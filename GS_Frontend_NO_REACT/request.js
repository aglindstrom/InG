// api call that loads in all the nodes
async function load_data(url, source){
    return await fetch(url)
                .then((resp) => resp.json())
                .then((data) => update_nodes(data, source))
                .catch((error) => console.error(error));
}

// api call to load keys potentially useful in the future for selecting the name column for the nodes
async function load_keys(){
    return await fetch('http://127.0.0.1:8000/api/graph/keys/')
                .then((resp) => resp.json())
                .then((data) => update_keys(data))
                .catch((error) => console.error(error));
}

// sources is an array of source objects 
//  source object looks like this
//  {port: ####, id: #}
//  where # represents a numeric digit
//
// action_name is the display name of the undo action

function request_graph_data(sources, action_name){
    let requests = [];
    sources.forEach((source) => {
        requests.push(load_data(get_url(source), source.id));
    });

    Promise.all(requests).then(() => {
        state_push(get_state(), action_name);
        unload_nodes();
        update_graph();
    });
}

async function request_neighbor_edges(node){
    hide_menu(node_drop_down);
    let params = new URLSearchParams({codes: node.name, depth: 1, direction: node.direction});
    let url = `${invokeURL}:${node.source.port}/api/graph?${params.toString()}`;
    return fetch(url).then((resp) => resp.json())
              .then((data) => update_group_links(data, node.name))
              .catch((error) => console.error(error));
}

function request_node_description(node){
    // use this to make node specific requests for extra node data from the extra node data server
    hide_menu(drop_down);
    let output = document.getElementById("graph-console");
    let params = new URLSearchParams({node:node});
    fetch(`${invokeURL}:${description_source.port}/node/description?${params.toString()}`)
        .then((resp) => resp.json())
        .then((desc) => output.innerHTML = ">" + desc.description);
}

function get_url(source){
    let request_selection = selected_nodes.filter((node) => source.id == node.group);
    let directions = request_selection.map((node) => node.direction);
    let selection = request_selection.map((node) => node.name);
    let params = new URLSearchParams({codes: selection.join(","), depth: depth, directions: directions.join(",")});
    let url = `${invokeURL}:${source.port}/api/graph?${params.toString()}`;
    return url;
}