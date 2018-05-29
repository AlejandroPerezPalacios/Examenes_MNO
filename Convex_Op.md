# Reporte: Convex Optimization for Big Data.

El objetivo del artículo es repasar los avances recientes en el campo de optimización convexa para big data, lo cual busca reducir los cuellos de botella computacionales de almacenamiento y comunicación. Los algoritmos están basados en principios simples y alcanzan una aceleración asombrosa incluso para problemas clásicos.

## Optimización convexa en el despertar del Big Data.

Aunque la optimización convexa data desde hace tiempo para el procesamiento de señales, su importancia en las formulaciones y optimización ha incrementado debido al surgimiento de nueva teoria de minimización de rangos o matrices ralas, además de los exitosos modelos estadísticos como máquina de soporte vectorial.

Hay múltiples razones por el aumento de interés en el campo, siend las dos más obvias la existencia de algortimos eficientes para calcular las soluciones óptimas y la abilidad de utilzar geometría convexa para probar propiedades útiles de la solución. 

Sin embargo, la popularidad por la optimización convexa pone a prueba a los diversos algoritmos para encontrar las soluciones óptimas que se basen en datos de gran tamaño provenientes de diversas fuentes como internet, texto u otros problemas que causan que el crecimiento sea de terabytes a exabytes cuando antes era de megabytes a gigabytes. Incluso con los avances en cuanto a paralelismo y cómputo distribuido la utilidad de los algoritmos clásicos no pasa más alla de la teoría para un conjunto de datos de tal tamaño.

Es por esto que la optimización convexa se tuvo que adaptar a este tipo de datos en los que una simple rutina de algebra lineal representa un problema. El contraste más claro con respecto a la optimización clásica es que no es necesario buscar una precisión cien por ciento certera ya que los problemas con este tipo de datos son simple e incluso inexactos.

## Idea general.

Una idea básica para entender los algoritmos de optimización para big data se basa en 3 pilares:

  + **Métodos de Primer orden**: Estos métodos obtienen una precisión baja/mediana de la solución númerica de su función objetivo, éstos métodos presentan una casi independencia a la dimensión del problema en cuanto a su convergencia y típicamente dependen de primitivas computacionales que son ideales para el cómputo distrbuido y paralelización.
  + **Aleatorización**: Las técnicas de aleatorización sobresalen en el ámbito de aproximación con respecto a otras técnicas ya que funcionan como un motor para los métodos de primer orden y además podemos controlar su comportamiento esperado.
  + **Cómputo distribuido y en paralelo**: Los métodos de primer orden son lo suficientemente flexibles para su implementación en paralelo o de manera distribuida.
  
Los 3 pilares se complementan de tal manera que ofrecen beneficios de escalabilidad para la optimización en big data. Por ejemplo los métodos de primer orden aleatorizados presentan una aceleración en convergencia con respecto a su contraparte determinística; en particular, un ejemplo que relaciono con esto de la maestría es el descenso gradiente estocástico para redes neuronales, ya que se desea minimizar una función pero por la cantidad de datos se van tomando muestras con las que se calcula el gradiente y se va convergiendo a la misma solución que el gradiente determinista pero de manera más rápida (para muchos datos).

Uno de los problemas habituales dentro de la optimización es el de regresión lineal, en el que deseamos hallar los coeficientes que mejor ajusten a nuestros datos, este ajuste se traduce en la minimimización del error cuadrático medio y recibe el nombre de mínimos cuadrados. Este problema es ampliamente usado en varios campos de la ciencia y en problemas prácticos y una de las maneras más claras de encontrar la solución es mediante descenso gradiente.

Se muestra en el artículo que el algortimo de descenso gradiente estocástico presenta una convergencia rápida con menos cantidad de datos que su contraparte determinística.




