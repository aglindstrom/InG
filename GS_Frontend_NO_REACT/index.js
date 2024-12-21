export default function load_data(){
  let _data = null;

  fetch("http://127.0.0.1:8000/api/graph/js_deps")
  .then((resp) => resp.json())
  .then((data) => _data = data);

  return _data;
}

export function main(){
  return `<div> hello world</div>`;
}

