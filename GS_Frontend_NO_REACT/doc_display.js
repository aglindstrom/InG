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
        doc_span.appendChild(document.createTextNode(doc));

        doc_display.appendChild(title_span);
        doc_display.appendChild(document.createElement('br'));
        doc_display.appendChild(doc_span);

        return extract_entities(doc_split)
            .then((entities) => {
                let out = {};
                let edges = [];
                const codes = entities.map((entity) => encode_entity(entity));
                highlight(doc_span, entities);
                Promise.all(codes)
                    .then((codes) => {
                        edges = create_entity_edges(codes);
                        out = {nodes: [... new Set(codes.map((code) => [code.entity, ...code?.data]).flat(Infinity))].map((node) => ({dx10: node})),
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
    const names = entities.map((entity) => entity.split(':').at(-1));
    element.textContent = "";
    names.forEach((name, idx) => {
        let split = doc_text.split(name, 2);
        doc_text = split.at(-1);

        let word_span = document.createElement('span');
        word_span.classList.add('word');
        word_span.textContent = split[0];
        element.appendChild(word_span);

        let name_span = document.createElement('span');
        name_span.classList.add(...['word', 'active']);
        name_span.title = entities[idx];
        name_span.textContent = name;
        element.appendChild(name_span);
    });
}


async function extract_entities(data){
    const url = `https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/VT-NE/VT:${data.join('%20')}`;
    console.log(url);

    let data_out = await fetch(url)
        .then((resp) => resp.json())
        .then((data) => data)
        .catch((error) => console.error("Extract_Entities ", error));

    return data_out;
}


async function encode_entity(entity){
    const url = `https://olive.is.mediocreatbest.xyz/4YCABK9FR0/api/v1/NE-DX/${entity.split(' ').join('%20')}?topk=20`;
    console.log(url);

    let data_out = await fetch(url)
        .then((resp) => resp.json())
        .then((data) => ({entity: entity, data: data.map((entry) => entry.slice(3, 7))}))
        .catch((error) => console.error("Encode_Entity ", error));

    return data_out;
}

function create_entity_edges(entities){
    return entities.map((entity) => entity.data?.map((code) => ({source: entity.entity, target: code, perplexity: '0', group: 0})));
}

function select_entity_nodes(entities){
    entities.forEach((e) => select_node({name: e.node.name, direction: e.node.direction, group: e.node.group}));
}