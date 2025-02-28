param(
  [string]$nodes = './data/ICD10CM,length=4.csv.gz',
  [string]$edges = './data/DgPg-GenIndFwd,top_p=0.1.csv.gz',
  [string]$names = 'dx10',
  [string]$port = '8000'
)

$ENV:ING_NODES=$nodes
$ENV:ING_EDGES=$edges
$ENV:ING_NAMES=$names

Write-Output $ENV:ING_NODES
Write-Output $ENV:ING_EDGES
Write-Output $ENV:ING_NAMES

fastapi dev --port $port ./src/main.py 
