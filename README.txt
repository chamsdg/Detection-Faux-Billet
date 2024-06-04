L’Organisation nationale de lutte contre le faux-monnayage, ou ONCFM,est une organisation publique ayant pour objectif de mettre en place des
méthodes d’identification des contrefaçons des billets en euros. Dans le cadre de cette lutte, nous souhaitons mettre en place un algorithme qui
soit capable de différencier automatiquement les vrais des faux billets

Objectifs
Lorsqu’un billet arrive, nous avons une machine qui consigne l’ensemble de ses caractéristiques géométriques. Au travers de nos années de lutte,
nous avons observé des différences de dimensions entre les vrais et les faux billets. Ces différences sont difficilement notables à l’œil nu, mais une
machine devrait sans problème arriver à les différencier. Ainsi, il faudrait construire un algorithme qui, à partir des caractéristiques
géométriques d’un billet, serait capable de définir si ce dernier est un vrai ou un faux billet.

Fonctionnement général
Comme vu précédemment, nous avons à notre disposition six données géométriques pour chaque billet. L’algorithme devra donc être capable de
prendre en entrée un fichier contenant les dimensions de plusieurs billets, et de déterminer le type de chacun d’entre eux, à partir des seules
dimensions. Nous fournissons à ce sujet le format type de nos fichiers de billets avec lequel l’algorithme sera censé fonctionner, au sein d’un fichier
nommé billets_production.csv.

Nous aimerions pouvoir mettre en concurrence deux méthodes de
prédiction :

une régression logistique classique ;
un k-means, duquel seront utilisés les centroïdes pour réaliser la prédiction.
Cet algorithme se devra d’être naturellement le plus performant possible pour identifier un maximum de faux billets au sein de la masse de billets
analysés chaque jour.