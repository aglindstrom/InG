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
        let doc_split = reader.result.split(/\s+/);
        doc_split = doc_split.map((word) => {
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
    }

    reader.readAsText(files[0]);    
}



function drag_over_handler(event){
    event.preventDefault();
}