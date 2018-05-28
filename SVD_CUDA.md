# Reporte: Singular Value Decomposition on GPU using CUDA.

La factorización SVD es una técnica importante utilizada para la factorización de una matriz rectangular. Bajo SVD las
operaciones con matrices son más robustas por lo que se utiliza para calcular la pesudo-inversa de una matriz, resolver sistemas de ecuaciones lineales u optimización de mínimos cuadrados, entre otros.

En principio la factorización SVD de una matriz **A** está definida por:
<a href="http://www.codecogs.com/eqnedit.php?latex=A=U&space;\Sigma&space;V^T" target="_blank"><img src="http://latex.codecogs.com/gif.latex?A=U&space;\Sigma&space;V^T" title="A=U \Sigma V^T" /></a>
donde <a href="http://www.codecogs.com/eqnedit.php?latex=U&space;\in&space;\mathbb{R}^{mxn},&space;V&space;\in&space;\mathbb{R}^{nxn},&space;\Sigma&space;\in&space;\mathbb{R}^{mxn}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?U&space;\in&space;\mathbb{R}^{mxn},&space;V&space;\in&space;\mathbb{R}^{nxn},&space;\Sigma&space;\in&space;\mathbb{R}^{mxn}" title="U \in \mathbb{R}^{mxn}, V \in \mathbb{R}^{nxn}, \Sigma \in \mathbb{R}^{mxn}" /></a> matrices
ortognales y <a href="http://www.codecogs.com/eqnedit.php?latex=\Sigma" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Sigma" title="\Sigma" /></a> matriz diagonal con <a href="http://www.codecogs.com/eqnedit.php?latex=s_{ii}>=0" target="_blank"><img src="http://latex.codecogs.com/gif.latex?s_{ii}>=0" title="s_{ii}>=0" /></a> con orden descendiente.

El objetivo del artículo _Singular Value Decomposotion on GPU using CUDA_ es realizar esta factorización en la GPU aprovechando el increíble poder de cómputo que la caracteriza. El rápido crecimiento 
en desempeño del hardware de gráficos llevó a la GPU a ser un fuerte candidato para realizar múltiples operaciones de caracter intensivo
y en paralelo. Es por esto que han surgido diversos ambientes que soporten la programación en GPU, NVIDIA provee CUDA como un ambiente operacional
con un lenguaje tipo C para los procesadores programables.

Este artículo expone que aunque el cómputo en la GPU se ha aprovechado para el cómputo científico no lo ha sido para la factorización SVD, problema
que por sus múltiples aplicaciones se ve necesario realizar dicha implementación.

La factorización SVD se puede obtener por diversos algortimos, en este artículo se expone el de Golub-Reinsch (Bidiagonalización y diagonalización) dado que es simple y compacto además
de mapear la arquitectura SIMD a la GPU. En resumen este algortimo realiza lo siguiente:

  1. La matriz A se reduce a una matriz bidiagonal usando una serie de transformaciones HouseHolder.
  2. La matriz reducida se diagonaliza utilizando iteraciones QR alternadas implícitamente.

En notación matemática se resume de la siguiente manera:

  1. <a href="http://www.codecogs.com/eqnedit.php?latex=B&space;\leftarrow&space;Q^TAP" target="_blank"><img src="http://latex.codecogs.com/gif.latex?B&space;\leftarrow&space;Q^TAP" title="B \leftarrow Q^TAP" /></a> {Bidiagonalización de A a B}
  2. <a href="http://www.codecogs.com/eqnedit.php?latex=\Sigma&space;\leftarrow&space;X^TBY" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Sigma&space;\leftarrow&space;X^TBY" title="\Sigma \leftarrow X^TBY" /></a> {Diagonalización de B a <a href="http://www.codecogs.com/eqnedit.php?latex=\Sigma" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Sigma" title="\Sigma" /></a>}
  3. <a href="http://www.codecogs.com/eqnedit.php?latex=U&space;\leftarrow&space;QX" target="_blank"><img src="http://latex.codecogs.com/gif.latex?U&space;\leftarrow&space;QX" title="U \leftarrow QX" /></a>
  4. <a href="http://www.codecogs.com/eqnedit.php?latex=V^T&space;\leftarrow&space;(PY)^T" target="_blank"><img src="http://latex.codecogs.com/gif.latex?V^T&space;\leftarrow&space;(PY)^T" title="V^T \leftarrow (PY)^T" /></a> {Calcular matrices U,V^T ortogonales y SVD de <a href="http://www.codecogs.com/eqnedit.php?latex=A=U&space;\Sigma&space;V^T" target="_blank"><img src="http://latex.codecogs.com/gif.latex?A=U&space;\Sigma&space;V^T" title="A=U \Sigma V^T" /></a>}

Es aquí cuando comienza lo interesante, debemos encontrar un algoritmo para poder realizar la bidiagonalización de A a B; para este caso en particular se
realiza una serie de transformaciones Householder unitarias que nos ayudarán a obtener una matriz con ceros en la i-ésima fila e i-ésima columna, por ejemplo para la primera iteración: Para una matriz A de mxn, tomamos <a href="http://www.codecogs.com/eqnedit.php?latex=u^{(1)}\in\mathbb{R}^m" target="_blank"><img src="http://latex.codecogs.com/gif.latex?u^{(1)}\in\mathbb{R}^m" title="u^{(1)}\in\mathbb{R}^m" /></a> para el vector A(1:m,1) y <a href="http://www.codecogs.com/eqnedit.php?latex=v^{(1)}\in\mathbb{R}^n" target="_blank"><img src="http://latex.codecogs.com/gif.latex?v^{(1)}\in\mathbb{R}^n" title="v^{(1)}\in\mathbb{R}^n" /></a> para A(1,2:n)
tal que:

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;\hat{A}&space;&=&space;(I-\sigma_{1,1}u^{(1)}{u^{(1)}}^T)\\&space;&=&space;H_1AG_1&space;=\begin{bmatrix}&space;\alpha_1&space;&&space;\beta_1&&space;0&&space;\cdots&&space;0\\&space;0&&space;x&&space;x&&space;\cdots&&space;x\\&space;\vdots&&space;x&&space;x&&space;\cdots&&space;x\\&space;0&&space;x&&space;x&&space;\cdots&&space;x&space;\end{bmatrix}&space;\end{align*}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{align*}&space;\hat{A}&space;&=&space;(I-\sigma_{1,1}u^{(1)}{u^{(1)}}^T)\\&space;&=&space;H_1AG_1&space;=\begin{bmatrix}&space;\alpha_1&space;&&space;\beta_1&&space;0&&space;\cdots&&space;0\\&space;0&&space;x&&space;x&&space;\cdots&&space;x\\&space;\vdots&&space;x&&space;x&&space;\cdots&&space;x\\&space;0&&space;x&&space;x&&space;\cdots&&space;x&space;\end{bmatrix}&space;\end{align*}" title="\begin{align*} \hat{A} &= (I-\sigma_{1,1}u^{(1)}{u^{(1)}}^T)\\ &= H_1AG_1 =\begin{bmatrix} \alpha_1 & \beta_1& 0& \cdots& 0\\ 0& x& x& \cdots& x\\ \vdots& x& x& \cdots& x\\ 0& x& x& \cdots& x \end{bmatrix} \end{align*}" /></a>
