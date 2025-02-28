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

// Make a request for graph data using each node in a list of nodes as anchors
// @params:
//  nodes: the list of nodes to make request with
//  action_name: name of the action in the undo stack

function request_graph_data(nodes, action_name){
    let requests = [];
    nodes.forEach((node) => {
        requests.push(load_data(get_url(node), node));
    });

    Promise.all(requests).then(() => {
        state_push(get_state(), action_name);
        unload_nodes();
        update_graph();
    });
}

async function request_neighbor_edges(node){
    let params = new URLSearchParams({codes: node.name, depth: 1, direction: node.direction});
    let url = `${invokeURL}:${node.source.port}/api/graph?${params.toString()}`;
    
    return fetch(url).then((resp) => resp.json())
              .then((data) => update_group_links(data, node.name))
              .catch((error) => console.error(error));
}

function request_node_description(node){
    // use this to make node specific requests for extra node data from the extra node data server
    let output = document.getElementById("graph-console");
    let params = new URLSearchParams({node:node});
    fetch(`${invokeURL}:${description_source.port}/node/description?${params.toString()}`)
        .then((resp) => resp.json())
        .then((desc) => output.innerHTML = ">" + desc.description);
}


// build a url to request a single node from the pre-defined server and port
// @params:
//  node: an Object that has a name, direction, and group

function get_url(node){
    let codes = node.name;
    let directions = node.direction;
    let params = new URLSearchParams({codes: codes, depth: depth, directions: directions});
    let url = `${invokeURL}:${graph_sources[node.group].port}/api/graph?${params.toString()}`;
    console.log(url);
    return url;
}