# Opdracht AS 2.2

IMAGE TRAININGSPROCES

De score aan het einde van het trainingsproces is ...
Hieruit kunnen we concluderen dat het model niet heel veel verbeterd is, maar wel consistent is geworden.
Een probleem die ik ben tegengekomen is met de decaying epsilon, en de input shape van (32,8) te laten werken met een predict shape van (1, 8). Dit komt omdat ik ook de hoeveelheid tensors had gedefinieerd in de opbouw van het model, terwijl je alleen de lengte van de tensor hoeft mee te geven.