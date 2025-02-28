let state_stack = [];

function state_push(state, display){
    state_stack.push({"state": state, "display": display});
    update_state_stack_element();
}

function state_pop(){
    state_stack.pop();
}

function stack_multi_pop(pops){
    for(let i = 0; i < pops; i++){
        state_pop();
    }
    update_state_stack_element();
    let state = JSON.parse(state_stack[state_stack.length-1]['state']);
    console.log(`${state_stack[state_stack.length-1]['display']}: `, state);
    set_state(state);
    console.log(JSON.parse(get_state()));
}

///////////// DOM INTERACTIONS ////////////////
function update_state_stack_element(){
    let state_stack_element = document.getElementById("state_stack");
    remove_children(state_stack_element);

    state_stack.slice(1).forEach((state, index) => {
        let state_element = state_make_element(index + ": " + state.display);
        state_stack_element.append(state_element);
    });
}

function state_make_element(display){
    let div = document.createElement("div");
    let content = document.createTextNode(display);
    
    div.classList.add("montserrat-body");
    div.appendChild(content);
    div.addEventListener("click", (e) => state_element_handle_click(e));
    
    return div;
}

function state_element_handle_click(e){
    let target = e.target;
    let parent = target.parentNode;
    let children = [...parent.childNodes];
    let index = (children.indexOf(target));

    stack_multi_pop(children.length - index);
}

////// Utility ///////

function remove_children(element){
    // from MDN documentation
    while(element.firstChild){
        element.removeChild(element.firstChild);
    }
}