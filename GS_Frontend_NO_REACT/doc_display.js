function drop_handler(event){
    const doc_display = document.getElementById('doc-display');
    const reader = new FileReader();
    const files = [];
    event.preventDefault();

    if(event.dataTransfer.items){
        [...event.dataTransfer.items].forEach((item) =>{
            if(item.kind === 'file' && item.getAsFile().type === 'text/plain'){
                files.push(item.getAsFile());
            }
        });
    }


    reader.onload = () => {
        while(doc_display.firstChild){
            doc_display.removeChild(doc_display.firstChild);
        }
        const doc = reader.result;
        const doc_split = doc.split(/\s+/)
        // create Header with filename
        let title_span = document.createElement('span');
        title_span.className += 'montserrat-title title-case';
        title_span.appendChild(document.createTextNode(files[0].name.split('.')[0]));

        let doc_span = document.createElement('span');
        doc_span.className += 'montserrat-body';
        doc_span.style.opacity = 0;
        doc_span.appendChild(document.createTextNode(doc));

        doc_display.appendChild(title_span);
        doc_display.appendChild(document.createElement('br'));
        doc_display.appendChild(doc_span);

        return extract_entities(doc)
            .then((entities) => {
                const req_ent = [...new Set(entities.filter((entity) => entity.split(':')[1] !== 'Clinical_event'))];
                console.log(req_ent);
                const code_requests = req_ent.map((entity, idx) => encode_entity(entity, idx));
                highlight(doc_span, entities);
                handle_code_requests(code_requests);
            });
    }

    reader.readAsText(files[0]);
}

async function handle_code_requests(code_requests){
    let out = {};
    let edges = [];
    while(code_requests.length){
        const code = await Promise.race(code_requests);
        for(const req of code_requests){
            req.then((res) => {if(res === code){code_requests.splice(code_requests.indexOf(req), 1); console.log(code);}})
               .catch((error) => {console.error("Code_Request: ", error); code_requests.splice(code_requests.indexOf(req), 1); console.log(code_requests.length); });
        }
        
        if(code === undefined || code?.error){
            continue;
        }else{
            const node = ({entity: {name: code.entity?.split(':'),
                              dx10: code.entity?.split(':').at(-1)},
                        data: code.data?.map((c) => ({
                                name: c.split(':'),
                                dx10: c.split(':').at(-1),
                                type: 'DX'
                            })
                        )
                    });
            edges = create_entity_edges(node);
            node.entity.fx = 0;
            out = {nodes: [... new Set([node.entity, ...node?.data]?.flat(Infinity))],
                   edges: [... new Set(edges?.flat(Infinity))]};
            if(!link_data['group_0']){ link_data['group_0'] = [];}
            link_data['group_0'].push(...out.edges);
            out.nodes = out.nodes.filter((node) => !loaded?.includes(node.dx10))
            loaded.push(...out.nodes.map((node) => node['dx10']));
            node_data.push(...out.nodes);
            update_graph();
        }
    }
}

function drag_over_handler(event){
    event.preventDefault();
}


function highlight(element, entities){
    let doc_text = element.textContent;
    let parent = element.parentElement;
    const names = entities.map((entity) => entity.split(':').at(-1));
    const delay = 75;
    let last = 0;
    parent.removeChild(element);
    element.textContent = "";
    names.forEach((name, idx) => {
        let expresion = new RegExp(`${name}(.*)`, 's');
        let split = doc_text.split(expresion,2);
        doc_text = split[1];

        let word_span = document.createElement('span');
        word_span.classList.add(...['word', 'start', 'montserrat-body']);
        word_span.textContent = split[0];
        word_span.style.animationDelay = `${(idx*2)*delay}ms`;
        word_span.addEventListener("animationend", (e) => { e.target.classList.add('end'); e.target.classList.remove('start')});
        parent.appendChild(word_span);

        let name_span = document.createElement('span');
        name_span.classList.add(...['word', 'active', 'start', 'montserrat-body']);
        name_span.title = entities[idx];
        name_span.textContent = name;
        name_span.style.animationDelay = `${(idx*2+1)*delay}ms`
        name_span.addEventListener("animationend", (e) => { e.target.classList.add('end'); e.target.classList.remove('start')});
        parent.appendChild(name_span);
    });

    if(doc_text){
        let word_span = document.createElement('span');
        word_span.classList.add(...['word', 'start', 'montserrat-body']);
        word_span.textContent = doc_text;
        word_span.style.animationDelay = `${(names.length*2)*delay}ms`;
        word_span.addEventListener("animationend", (e) => { e.target.classList.add('end'); e.target.classList.remove('start')});
        parent.appendChild(word_span);
    }
}


async function extract_entities(data){
    const segmenter = new Intl.Segmenter("en-US", {granularity:"sentence"});
    const sentences = segmenter.segment(data)[Symbol.iterator]();

    const requests = sentences.map(async (sentence) => {
        let url = new URL(`https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/VT-NE/`);

        let resp = await fetch(url, {
                method: "POST",
                headers: {
                    'Content-Type':'application/json'
                },
                body: `{\"data\":\"VT:${sentence.segment.split(/\s+/).join(' ')}\"}`
            }).catch((error) => {console.error("Extract_Entities ", error)})

        const data = await resp.json();

        if(!resp.ok){
            console.error(data, `sentence: ${sentence.segment}`);
        }
        
        return data;
    });

    let data_out = await Promise.allSettled(requests);
    data_out = data_out.map((a) => (a.status === 'fulfilled'? (a.value.detail? [] : a.value) : undefined)).flat();
    return data_out;
}

async function encode_entity(entity, idx){
    const url = `https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/NE-DX/${entity.split(/\s+/).join('%20')}?topk=${topk}`;
    let data_out = await fetch(url)
        .then((resp) => resp.json())
        .then((data) => ({entity: entity, data: data.map((node) => node.padEnd(7, '-')), loc:idx}))
        .catch((error) => console.error("Encode_Entity ", error));

    return data_out;
}

function create_entity_edges(entity){
    return entity.data?.map((code) => ({source: entity.entity.dx10, target: code.dx10, perplexity: '0', group: 0}));
}

function select_entity_nodes(entities){
    entities.forEach((e) => select_node({name: e.node.name, direction: e.node.direction, group: e.node.group, type: 'NE'}));
}