function open_user_menu(e){
    const x = e.target.offsetLeft + (e.target.offsetWidth / 2);
    const y = e.target.offsetTop + (e.target.offsetHeight);
    console.log(`(${x}, ${y})`);
    move_menu(application_drop_down, {pageX: x, pageY: y});
    show_menu(application_drop_down);
}