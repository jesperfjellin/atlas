# Atlas

Atlas is an experiment in learning how OpenStreetMap changes over time. It asks whether a small machine-learning model can predict future mapping activity and learn useful representations of places from their histories.

The first study focuses on Norway. Atlas divides the country into a fixed grid of cells and follows each cell month by month. The model will receive two years of observations and predict changes over the following six months. Cells remain part of the study even before anyone maps something in them.

OpenStreetMap history records changes to a map. Those changes can reflect new construction, delayed mapping, corrections, imports, or changing tagging conventions. Atlas studies that recorded history without assuming that an edit identifies when something changed in the physical world.

The experiment keeps three kinds of change separate: edits to map objects, additions or removals of mapped categories, and changes in the total mapped contents of a cell. Buildings, roads, points of interest, and land use provide the initial vocabulary. This distinction lets Atlas describe a building losing its classification, a road changing shape, or an object moving between cells without treating them as the same event.

Prediction is only part of the aim. Atlas will also learn compact representations, called embeddings, of places at different points in time. The question is whether these representations capture useful patterns: places with similar mapping histories, recurring kinds of change, or trajectories that simple summaries miss.

Success requires evidence. Predictions must improve on sensible simple baselines, and the learned representations must add value beyond a basic compression of the same inputs. Evaluation uses future periods and geographic areas kept separate from training. A model that adds no useful signal is a valid result too.

Atlas is a small proof of concept. The first Norway history dataset is prepared; predictive skill and useful learned representations have not yet been demonstrated. The [specification](SPEC.md) defines the experiment, and the [roadmap](PROGRESS.md) records what works and what remains.

OSM data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the Open Database License (ODbL). The study boundary comes from [Natural Earth](https://www.naturalearthdata.com/).
