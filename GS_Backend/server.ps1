param(
  [string]$nodes = './data/ICD10CM,length=4.csv.gz',
  [string]$edges = './data/DgPg-GenIndFwd,top_p=0.1.csv.gz',
  [string]$names = 'dx10',
  [string]$port = '8000'
)

$ENV:ING_NODES=$nodes
$ENV:ING_EDGES=$edges
$ENV:ING_NAMES=$names

echo "NODE FILE: " $ENV:ING_NODES
echo "EDGE FILE: " $ENV:ING_EDGES
echo "COLU NAME: " $ENV:ING_NAMES
echo "SERV PORT: " $port 

fastapi dev --port $port ./src/main.py 
