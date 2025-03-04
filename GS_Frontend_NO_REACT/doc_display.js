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
        const doc_split = doc.split(/\s+/).map((word) => {
            let span = document.createElement('span');
                span.appendChild(document.createTextNode(word));
                span.classList += "word";
            return span;
        });
        // create Header with filename
        let span = document.createElement('span');
        span.className += 'montserrat-title title-case';
        span.appendChild(document.createTextNode(files[0].name.split('.')[0]));
        doc_display.appendChild(span);

        // add document text
        doc_display.appendChild(document.createElement('br'));
        doc_split.forEach((node) => {
            doc_display.appendChild(document.createElement('br'));
            doc_display.appendChild(node);
        });

        const entities = extract_entities(doc);
        highlight(doc_display, entities);
        select_entity_nodes(entities);
    }

    reader.readAsText(files[0]);
}


function drag_over_handler(event){
    event.preventDefault();
}

function highlight(element, entities){
    const children = [...element.children];
    console.log(entities);
    const words = children.filter((child) => 
                  child.nodeName.toLowerCase() === 'span' && 
                  child.className.includes('word'));

    const isEntity = (word) => {
        const idx = entities.indexOf(...entities.filter((e) => e.entity === word));
        return idx;
    }

    words.forEach((word) => {
        const idx = isEntity(word.textContent);
        if(idx != -1){
            const title = `entity node: ${entities[idx].node.name}`
            word.classList.add('active');
            word.title = title;
        }
    });
}

function extract_entities(data){
    // TODO: Fetch from entity extractor 

    //PLACEHOLDER//
    const group = 1;
    const direction = 1;
    const data_s = data.split(/\s+/);
    const data_f = data_s.filter(() => Math.random() > .75);
    const data_out = data_f.map((d, idx) => ({entity: d, node:{name: idx, direction:direction, group:group}}));
    return data_out;
}

function select_entity_nodes(entities){

    entities.forEach((e) => select_node({name: e.node.name, direction: e.node.direction, group: e.node.group}));
}