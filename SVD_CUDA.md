# Reporte: Singular Value Decomposition on GPU using CUDA.

La factorización SVD es una técnica importante utilizada para la factorización de una matriz rectangular. Bajo SVD las
operaciones con matrices son más robustas por lo que se utiliza para calcular la pesudo-inversa de una matriz, resolver sistemas
de ecuaciones lineales u optimización de mínimos cuadrados, entre otros.

En principio la factorización SVD de una matriz **A** está definida por:
<a href="http://www.codecogs.com/eqnedit.php?latex=A=U&space;\Sigma&space;V^T" target="_blank"><img src="http://latex.codecogs.com/gif.latex?A=U&space;\Sigma&space;V^T" title="A=U \Sigma V^T" /></a>
donde <a href="http://www.codecogs.com/eqnedit.php?latex=U&space;\in&space;\mathbb{R}^{mxn},&space;V&space;\in&space;\mathbb{R}^{nxn},&space;\Sigma&space;\in&space;\mathbb{R}^{mxn}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?U&space;\in&space;\mathbb{R}^{mxn},&space;V&space;\in&space;\mathbb{R}^{nxn},&space;\Sigma&space;\in&space;\mathbb{R}^{mxn}" title="U \in \mathbb{R}^{mxn}, V \in \mathbb{R}^{nxn}, \Sigma \in \mathbb{R}^{mxn}" /></a> matrices
ortognales y diagonal.
