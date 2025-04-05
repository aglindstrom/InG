function drop_handler(event){
    const doc_display = document.getElementById('doc-display');
    const reader = new FileReader();
    const files = [];

    console.log("Drop Event: ", event);
    event.preventDefault();

    if(event.dataTransfer.items){
        [...event.dataTransfer.items].forEach((item) =>{
            console.log('Type: ', item.getAsFile().type);
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
                let out = {};
                let edges = [];
                const req_ent = entities.filter((entity) => entity.split(':')[1] !== 'Clinical_event');
                const code_requests = req_ent.map((entity) => encode_entity(entity));
                highlight(doc_span, entities);
                Promise.allSettled(code_requests)
                    .then((code_list) => {
                        let codes = [...code_list.map((code) => code.status === 'fulfilled'? code.value: []).flat(Infinity)];
                        //state_push(get_state(), 'GET ENTITIES');
                        codes = codes.map((code) => ({entity: {name: code.entity.split(':'), dx10: code.entity.split(':').at(-1)}, 
                                                      data: code.data.map((node) => ({name: node.split(':'), dx10: node.split(':').at(-1)})) }) );
                        edges = create_entity_edges(codes);
                        out = {nodes: [... new Set(codes.map((code) => [code.entity, ...code?.data]).flat(Infinity))],
                               edges: [... new Set(edges.flat(Infinity))]};
                        if(!link_data['group_0']){ link_data['group_0'] = [];}
                        link_data['group_0'].push(...out.edges);
                        node_data.push(...out.nodes);
                        loaded.push(...out.nodes.map((node) => node['dx10']));
                        update_graph();
                    });
            });
    }

    reader.readAsText(files[0]);
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
        let url = new URL(`https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/VT-NE`);

        let resp = await fetch(url, {
                method: "POST",
                headers: {
                    'Content-Type':'application/json'
                },
                body: `{\"data\":\"VT:${sentence.segment}\"}`
            }).catch((error) => {console.error("Extract_Entities ", error)})

        const data = await resp.json();

        if(!resp.ok){
            console.log(data, `sentence: ${sentence.segment}`);
        }
        
        return data;
    });

    let data_out = await Promise.allSettled(requests);
    data_out = data_out.map((a) => (a.status === 'fulfilled'? (a.value.detail? [] : a.value) : undefined)).flat();
    console.log(data_out);
    return data_out;
}

async function encode_entity(entity){
    const url = `https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/NE-DX/${entity.split(' ').join('%20')}?topk=${topk}`;
    console.log(url);

    let data_out = await fetch(url)
        .then((resp) => resp.json())
        .then((data) => ({entity: entity, data: data.map((node) => node.padEnd(7, '-'))}))
        .catch((error) => console.error("Encode_Entity ", error));

    return data_out;
}

function create_entity_edges(entities){
    return entities.map((entity) => entity.data?.map((code) => ({source: entity.entity.dx10, target: code.dx10, perplexity: '0', group: 0})));
}

function select_entity_nodes(entities){
    entities.forEach((e) => select_node({name: e.node.name, direction: e.node.direction, group: e.node.group}));
}