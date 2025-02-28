
    // constants
    const context = document.getElementById("graph");
    const width = context.scrollWidth;   // working width and height for the simulation
    const height = context.scrollHeight; //
    const padding = 20;
    const drop_down = document.getElementById("drop-down");           // the dropdown menu for right click
    const node_drop_down = document.getElementById("node-drop-down"); // the dropdown menu for node clicks
    const menus = [drop_down, node_drop_down];
    const color = d3.scaleSequential(d3.interpolateTurbo);        // node colors for comparison string
    const link_color = d3.scaleSequential(d3.schemeSet1);         // node colors for comparison string
    const invokeURL = "http://127.0.0.1";                         // the address of the API
    const graph_sources = [{port:8000, id:0},                     // list of valid source ports 
                           {port:8001, id:1}];                    //
    const description_source = {port:7999, id:'nodes'};           // list of valid description source ports
    
    // initial values
    let depth = document.getElementById("depth").value;                   // initial depth = 1
    let comparison = document.getElementById("comparison").value;         // initial comparison string = ''
    let name = 'dx10';                                                    // node name column. In the future a menu using the keys should be used to determin this
    let distance_threshold = document.getElementById("threshold").value;  // initial distance threshold
    let anchor_color = 'linear-gradient(90deg, rgb(255,133,0) 0%, rgba(75,75,75) 100%)';     // initial anchor color
    let node_color = document.getElementById("node_color").value;
    let scale = document.getElementById("scale").value;
    let coloring = 0;
    let labeling = 0;
    let clicked_node;                                                     // used for right click handling. Needed by more than one function so its a global

    let links = {};

    // graph data
    let loaded = [];      // list of node names that are currently loaded 
    let node_data = [];   // all the nodes
    let link_data = {};   // all the links
    let grouping_links = [];
    let selected_nodes = [];

    const init_state = get_state();
    state_push(init_state, "INITIALIZE");

    // Create SVG
    const svg = d3.select("#graph").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-width/2, -height/2, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    // Create force simulation
    const simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d[name]))
        .force("grouping", d3.forceLink().id(d => d[name]))
        .force("charge", d3.forceManyBody().strength(-150))
        .force("y_sort-", d3.forceY(-height/2).strength(d => .05*d.equiv))
        .force("y_sort+", d3.forceY(height/2).strength(d => .05*(1-d.equiv)))

    // make the arrow marker to indicate edge direction
    svg.append("defs")
       .append("marker")
       .attr("id", "arrow")
       .attr("viewBox", [0, 0, 10, 10])
       .attr("refX", -40)
       .attr("refY", 5)
       .attr("markerWidth", 4)
       .attr("markerHeight", 4)
       .attr("orient", "auto")
       .append("path")
       .attr("cx", 0)
       .attr("cy", 0)
       .attr("d", d3.line()([[0, 0], [10, 5], [0, 10]]));
       

    // Add nodes labels and links to the graph
    let link = svg.append("g").selectAll(".link");
    let grouping = svg.append("g").selectAll(".grouping");
    let node = svg.append("g").selectAll(".node");
    let text = svg.append("g").selectAll(".label");

    //
    // updateGraph this is the d3 rendering and simulation part 
    //
    function update_graph() {
      // Filter nodes and links based on selectedNodeIds
      if(!node_data) return;

      let filtered_links = [];
      for(const group in link_data){
        filtered_links.push(link_data[group]);
      }
      filtered_links = filtered_links.flat();
      filtered_links = filtered_links.filter(d => (d.perplexity) > distance_threshold);

      // Update nodes
      node = node.data(node_data, d => d[name]);
      node.exit().remove();
      node = node.enter()
          .append("rect")
          .attr("class", "node")
          .attr("height", 14)
          .attr("width", 60)
          .attr("rx", 4)
          .on("click", e => node_click(e))
          .on("contextmenu", e => anchor_right_click(e))
          .call(drag(simulation))
          .merge(node);

      node.append("title")
          .text( d => d[name] );

      d3.selectAll("rect").attr("fill", d => { d.equiv = .5; //similarity(d);
            return (is_anchor(d[name]))? anchor_color: node_color;}); //color((1-d.equiv)*scale);});
      
      // Update lables
      text = text.data(node_data, d => d[name]);
      text.exit().remove();
      text = text.enter()
          .append("text")
          .attr("class", "label")
          .attr("text-anchor", "middle")
          .attr("dx", 0)
          .attr("dy", 3.5)
          .merge(text);
      text.text(d => get_label(d).slice(0,7));

      // Update links
      link = link.data(filtered_links, d => d.source[name] + "-" + d.target[name]);
      link.exit().remove();
      link = link.enter()
          .append("line")
          .attr("class", "link")
          .attr("marker-start", "url(#arrow)")
          .attr("stroke", d => link_color(d.group))
          .merge(link);

      link.append("title")
          .text(d => `group: ${d.group}`);

      grouping = grouping.data(grouping_links, d => d.source[name] + "-" + d.target[name]);
      grouping.exit().remove();
      grouping = grouping.enter()
          .append("line")
          .attr("class", "grouping")
          .attr("stroke", "#ffff00")
          .merge(grouping);
      
      let max_perplexity = filtered_links.reduce((acc, link) => acc > link.perplexity? acc: link.perplexity, 0);

      // Restart simulation
      simulation.nodes(node_data).on("tick", ticked);
      simulation.force("link").links(filtered_links).strength(.05); //d => max_perplexity/d.perplexity*scale);
      console.log("UPDATE GRAPH: ", grouping_links);
      simulation.force("grouping").links(grouping_links).strength(1);
      simulation.alpha(1).restart();

    }

    function ticked() {
        link.attr("x1", d => clamp(d.source.x, width/2, padding))
            .attr("y1", d => clamp(d.source.y, height/2, padding))
            .attr("x2", d => clamp(d.target.x, width/2, padding))
            .attr("y2", d => clamp(d.target.y, height/2, padding));

        node.attr("x", d => clamp(d.x, width/2, padding)-30)
            .attr("y", d => clamp(d.y, height/2, padding)-7);

        text.attr("x", d => clamp(d.x, width/2, padding))
            .attr("y", d => clamp(d.y, height/2, padding));
    }

    function drag(simulation) {
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = clamp(event.x, width/2, padding);
            d.fy = clamp(event.y, height/2, padding);
        }

        function dragged(event, d) {
            d.fx = clamp(event.x, width/2, padding);
            d.fy = clamp(event.y, height/2, padding);
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            if(is_anchor(d[name])){
              pin(d);
            }else{
              d.fx = null;
              d.fy = null;
            }
        }

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);      
    }
    // END updateGraph

    // called by load_data ONLY
    // loads new data into the node and link lists
    // also removes data from them
    function update_nodes(data, source){
      if(data == null) return;

      let nodes = JSON.parse(data.nodes);
      let edges = JSON.parse(data.edges);

      nodes.forEach((node) => {            // check if the node is loaded already 
        if(!loaded.includes(node[name])){  // if it is don't update it so d3 doesn't explode
          node_data.push(node);
          loaded.push(node[name]);
        }else{
          let a = node_data.filter(n => n[name] == node[name]);
          a['similarity'] = node['similarity'];
        }
      });

      edges.forEach((edge) => {
        if(edge['group']) return;
        edge['group'] = source;
      });

      // prune unused nodes
      node_data.map((node) => edges.filter((e) => e.source == node[name] || e.target == node[name])?node:undefined);
      node_data.flat();

      links[`group_${source}`] = JSON.parse(JSON.stringify(edges)); // keep a copy of the links
      link_data[`group_${source}`] = edges;                         // set the link data

      update_graph();
    }

    function unload_nodes(){
      let link_ids = [];
      let keep = [];

      for(const group in link_data){
        link_ids.push(link_data[group]);
      }

      link_ids = link_ids.flat();
      link_ids = link_ids.map((link) => [link.source.id ?? link.source, link.target.id ?? link.target]);
      link_ids = link_ids.flat();

      // console.log("before: ", node_data);
      node_data = node_data.filter((node) => link_ids.includes(node));
      // console.log("after : ", node_data);
    }

    // filters out any links with nodes that are not currently displayed
    function update_group_links(data){
      let new_links = JSON.parse(data.edges);
      new_links.forEach((link, index) => {
        let fl = node_data.filter((node) => node[name] == link.source || node[name] == link.target);
        if(fl.length > 1) return;
        
        new_links[index] = undefined;
      });
      
      new_links = new_links.filter((node) => node !== undefined);

      console.log(new_links);
      grouping_links = new_links;
      hide_menu(node_drop_down);
      update_graph();
    }

    // updates the depth of the edge search
    function update_depth(){
      depth = document.getElementById("depth").value;

      request_graph_data(graph_sources, "UPDATE_DEPTH")
      update_graph();
    }

    // updates the edge weight threshold
    function update_threshold(){
      distance_threshold = document.getElementById("threshold").value;

      state_push(get_state(), "UPDATE THRESHOLD");
      update_graph();
    }

    function update_scale(){
      scale = document.getElementById("scale").value;

      state_push(get_state(), "UPDATE SCALE");
      update_graph();
    }

    // updates the selection of anchor nodes.
    // it also manages the directions list.
    function update_selection(){

      let new_selection = document.getElementById('dx10').value;
      console.log(new_selection);
      if(new_selection == ''){ // if the selection is empty re-initialize everything
        loaded = [];
        node_data = [];
        link_data = [];
        selected_nodes = [];
        
        console.log("selected_nodes: ", selected_nodes);
        state_push(get_state(), "UPDATE SELECTION");
        update_graph(); // update the graph so everything disapears
        return;
      }
      
      new_selection = new_selection.split(",").map((id) => id.trim().padEnd(4,'-'));

      document.getElementById('dx10').value = new_selection;

      let diff = selected_nodes.length - new_selection.length;
      
      // BUG HERE maybe?
      
      if(diff < 0){ // check if a node was added to the list
        for(let i = 0, j = 0; diff != 0; i++, j++){
          if((selected_nodes[i] ?? {name:''}).name  == new_selection[j]){ continue; }
          selected_nodes.splice(j, 0, {name: new_selection[j],
                                  direction: 1,
                                  group: 0,});
          diff++;
          j++;
        }
      }else{ // check if a node was deleted from the list
        for(let i = 0, j = 0; diff != 0; i++, j++){
          if((selected_nodes[i] ?? {name:''}).name == new_selection[j] ?? ''){ continue; }
          selected_nodes.splice(j, 1);
          diff--;
          j--;
        }
      }

      console.log("selected_nodes: ", selected_nodes);
      request_graph_data(graph_sources, "UPDATE_SELECTION");
    }

    // updates the color of the root nodes
    function update_anchor_color(){
      anchor_color = document.getElementById("anchor_color").value;
      state_push(get_state(), "UPDATE COLOR");
      update_graph();
    }

    function update_node_color(){
      node_color = document.getElementById("node_color").value;
      state_push(get_state(), "UPDATE COLOR");
      update_graph();
    }

    function update_coloring(mode){
      coloring = mode;

      state_push(get_state(), "UPDATE COLORING");
      update_graph();
    }

    function update_labeling(mode){
      labeling = mode;

      state_push(get_state(), "UPDATE COMPARISON");
      update_graph();
    }

    function get_label(node){
      return labeling == 0? node['dx10'] : Number.parseFloat(node['equiv']).toFixed(4);
    }

    // updates the string to compare with on each node
    function update_comparison(){
      comparison = document.getElementById("comparison").value;
      update_embedding_string();

      if(coloring == 1){
        update_nodes(get_url(8000));
      }

      state_push(get_state(), "UPDATE COMPARISON");
      update_graph();
    }

    function update_embedding_string(){
      let url = `${invokeURL}/api/embed`;
      const request = new Request(url, {
          headers: {'accept': 'application/json', 'Content-Type': 'application/json'},
          method:"POST", 
          body: JSON.stringify({"prompt": comparison}),
        });

      const post = fetch(request).then((resp) => resp.json() ).then((json) => console.error(json.detail));
    }


    // ratio of the number of characters in the string that match in order
    function similarity(node){
      return coloring == 0? string_ratio(comparison, node['desc']): cosine_similarity(node);
    }

    function cosine_similarity(node){
      return node['similarity'] ?? 0.0;
    }
    
    function string_ratio(a, b){
      return 0;
      if(!a.length || !b.length) return 0;

      const min_len = a.length < b.length ? a.length: b.length;
      const max_len = a.length > b.length ? a.length: b.length;
      let i = 0;

      for(i = 0; i < min_len; i++){
        if(a[i] !== b[i]){
          break;
        }
      }

      return (i/max_len);
    }

    // node click handler for left click (makes the node an anchor node)
    function node_click(e){
      hide_menus();
      let node_name = e.srcElement.__data__[name];
      clicked_node = e.srcElement.__data__;
      if(is_anchor(node_name)) return;

      move_menu(node_drop_down, e);
      show_menu(node_drop_down);
    }

    function add_anchor(group){
      selected_nodes.push({name: clicked_node[name], direction: 1, group: group});
      document.getElementById('dx10').value = selected_nodes.map((node) => node.name).join(',');
      update_selection();
      hide_menu(node_drop_down);
    }

    // node click handler for right click
    function anchor_right_click(e){
      hide_menus();
      clicked_node = e.srcElement.__data__;

      if(!is_anchor(clicked_node[name])) return;

      e.preventDefault(); // disable the normal context menu

      move_menu(drop_down, e); // move the dropdown into position
      show_menu(drop_down); // make the dropdown visible
    }
    
    function hide_menus(){
      menus.forEach((menu) => {
        hide_menu(menu)
      });
    }

    // hides the right click menu
    function hide_menu(menu){
      console.log("hiding: ", menu);
      menu.style.display = 'none';
    }

    function show_menu(menu){
      console.log("showing: ", menu);
      menu.style.display = 'block';
    }

    function move_menu(menu, e){
      if(e.pageY < height/2){
        menu.style.top = e.pageY + "px";
      }else{
        menu.style.top = (e.pageY + menu.offsetHeight) + "px";
      }

      menu.style.left = e.pageX + "px"; // move the dropdown   
    }

    function is_menu_visible(menu){
      return (menu.style.display != 'none');
    }

    // sets to search for edges where this node is the source and target
    function set_all_edges(){
      let node = selected_nodes.find((n) => n.name == clicked_node[name]);
      node.direction = 0;
      request_graph_data(graph_sources, `SET_ALL_EDGES: ${clicked_node[name]}`);
      hide_menu(drop_down);
    }

    // sets to search for edges where this node is the source
    function set_out_edges(){
      let node = selected_nodes.find((n) => n.name == clicked_node[name]);
      node.direction = 1;
      request_graph_data(graph_sources, `SET_OUT_EDGES: ${clicked_node[name]}`);
      hide_menu(drop_down);
    }

    // sets to search for edges where this node is the target
    function set_in_edges(){
      let node = selected_nodes.find((n) => n.name == clicked_node[name]);
      node.direction = 2;
      request_graph_data(graph_sources, `SET_IN_EDGES: ${clicked_node[name]}`);
      hide_menu(drop_down);
    }

    function unpin(node){
      if(!node['pinned']) return;
      node['fx'] = null;
      node['fy'] = null;
      node['pinned'] = false;
      state_push(get_state(), `UNPIN: ${node[name]}`);
      hide_menu(drop_down);
    }

    function pin(node){
      if(node['pinned']) return;
      node['fx'] = node['x'];
      node['fy'] = node['y'];
      node['pinned'] = true;
      state_push(get_state(), `PIN: ${node[name]}`);
      hide_menu(drop_down);
    }

    function change_server(e_node){
      let node = selected_nodes.filter((n) => e_node[name] == n.name)[0];
      console.log(node);
      if(node.group == 1){
        node.group = 0;
      }else if(node.group == 0){
        node.group = 1;
      }
      request_graph_data(graph_sources, `CHANGE SERVER: ${node.name}`);
      hide_menu(drop_down);
    }
    
    function clamp(val, bound, pad){
      return ((val < bound-pad)? ((val > -bound+pad)? val : -bound+pad): bound-pad);
    }

    function is_anchor(name){
      return selected_nodes.filter((node) => node.name == name).length != 0;
    }

///////////////// STATE FUNCS //////////////////

    function get_state(){
      const state = {
        "links": links,
        "loaded": loaded,
        "selected_nodes": selected_nodes,
        "node_data": node_data,
        "depth": depth,
        "comparison": comparison,
        "name": name,
        "distance_threshold": distance_threshold,
        "anchor_color": anchor_color,
        "scale": scale,
        "coloring": coloring,
        "labeling": labeling,
        "clicked_node": clicked_node
      };
      return JSON.stringify(state);
    }
    
    function set_state(state){
      loaded = state.loaded;
      selected_nodes = state.selected_nodes;
      node_data = state.node_data;
      link_data = state.links;
      depth = state.depth;
      comparison = state.comparison;
      name = state.name;
      distance_threshold = state.distance_threshold;
      anchor_color = state.anchor_color;
      scale = state.scale;
      coloring = state.coloring;
      labeling = state.labeling;
      clicked_node = state.clicked_node;

      document.getElementById('dx10').value = selected_nodes.map((node) => node.name).join(",");
      document.getElementById('depth').value = depth;
      document.getElementById('comparison').value = comparison;
      document.getElementById('threshold').value = distance_threshold;
      document.getElementById('anchor_color').value = anchor_color;
      document.getElementById('scale').value = scale;
      update_graph();
    }

/*
    let depth = document.getElementById("depth").value;                   // initial depth = 1
    let comparison = document.getElementById("comparison").value;         // initial comparison string = ''
    let name = 'dx10';                                                    // node name column. In the future a menu using the keys should be used to determin this
    let distance_threshold = document.getElementById("threshold").value;  // initial distance threshold
    let anchor_color = document.getElementById("anchor_color").value;     // initial anchor color
    let scale = document.getElementById("scale").value;
    let coloring = 0;
    let labeling = 0;
    let clicked_node;                                                     // used for right click handling. Needed by more than one function so its a global
*/

// this is a block of things that could be useful in the future
/*
https://observablehq.com/@mbostock/ramp
function ramp(color, n = 512) {
  const canvas = DOM.canvas(n, 1);
  const context = canvas.getContext("2d");
  canvas.style.margin = "0 -14px";
  canvas.style.width = "calc(100% + 28px)";
  canvas.style.height = "40px";
  canvas.style.imageRendering = "-moz-crisp-edges";
  canvas.style.imageRendering = "pixelated";
  for (let i = 0; i < n; ++i) {
    context.fillStyle = color(i / (n - 1));
    context.fillRect(i, 0, 1, 1);
  }
  return canvas;
}



    function update_keys(data){
      keys = data;
      if(keys == null) return;

      Object.keys(keys).forEach((key) =>{ 
        let option = nodeOptions.append("div")
          .text(key + ": ");

        let selector = {};

        switch(keys[key]["type"]){
          case "int64": 
            selector = option.append("select")
                                 .attr("id", `${key}-op`)
                                 .on("change", () => {update_path_params(key)});

            selector.append("option")
                    .attr("value", ">")
                    .text(">");
            selector.append("option")
                    .attr("value", "=")
                    .text("=");
            selector.append("option")
                    .attr("value", "<")
                    .text("<");

		        option.append("input")
		          .attr("type", "number")
              .attr("class", "number")
              .attr("placeholder", "number")
              .attr("min", keys[key]["min"])
              .attr("max", keys[key]["max"])
              .attr("id", key)
              .on("change", () => {update_path_params(key)});

            option.append("input")
                  .attr("type", "radio")
                  .attr("name", "isName")
                  .attr("value", key);
            break;

          case "object":
              selector = option.append("select")
                                 .attr("id", `${key}-op`)
                                 .on("change", () => {update_path_params(key)});
            selector.append("option")
                    .attr("value", ">")
                    .text(">");
            selector.append("option")
                    .attr("value", "=")
                    .text("=");
            selector.append("option")
                    .attr("value", "<")
                    .text("<");

		        option.append("input")
		          .attr("type", "text")
              .attr("class", "number")
              .attr("placeholder", "text")
              .attr("min", keys[key]["min"])
              .attr("max", keys[key]["max"])
              .attr("id", key)
              .on("change", () => {update_path_params(key)});

            option.append("input")
                  .attr("type", "radio")
                  .attr("name", "isName")
                  .attr("value", key)
                  .on("click", () => {update_name()});
            break;
        }
        });
      nodeOptions.append("input")
                 .attr("id", "update_graph")
                 .attr("type", "button")
                 .attr("value", "update_graph")
                 .on("click", () => {load_data(get_url(8000), 0)});
    }
*/
    

/*    function update_name(){
      let radio_buttons = document.getElementsByName("isName");

      for(const button of radio_buttons){
        if(button.checked){
          name = button.value;
          break;
        }
      }
      console.log("name: ", name);
      update_nodes(null);
    }

    function update_path_params(key){
      path_params[key] = {};
      path_params[key]['val'] = document.getElementById(key).value;
      path_params[key]['op'] = document.getElementById(`${key}-op`).value;
      console.log(key, path_params);
    }
*/

