# climate-change-crops
Attempting the [Future Crops challenge](https://www.kaggle.com/competitions/the-future-crop-challenge) on Kaggle - how well can we use data to predict crop yields under a changing climate?

The challenge here is interesting from a machine learning perspective: for each location, there is only one target data point per year, the end-of-season yield ($y$), but many features, such as temperature and moisture recordings for 240 days after sowing ($\textbf{x}$). Using only data from the past 39 years we need to predict crop yields for the next 50 years under projected climate change (this data is modelled).
<img width="810" height="529" alt="image" src="https://github.com/user-attachments/assets/f93b094d-1c7e-4aee-bf85-b37612b61c5a" />


## Unsupervised representation learning

Current Kaggle leaderboard attempts at have benefited from domain knowledge, using features such as [Growing Degree Days](https://en.wikipedia.org/wiki/Growing_degree-day) which has been used to predict plant growth for almost 300 years.
However, I am interested in giving myself a further challenge of using discovered feature engineering - an unsupervised representation learning.


## Resources and implemented ideas

### Are transformers all you need?
My first attempt was to simply implement a standard Transformer trained on data for each crop. How good of a sequence learner is it?
This was trained with input masking, inspired by Microsoft's work on world models in the context of gaming, which I talked to Sam Devlin about [(Kanervisto et al., 2025)](https://www.nature.com/articles/s41586-025-08600-3).
So the transformer doesn't only predict the crop yield, but also does this via a world model that can predict the climate data input.
This performs surprisingly poorly, despite training with dropout [(Srivastava et al., 2014)](https://www.nature.com/articles/s41586-025-08600-3) and using RevIn [(Kim et al., 2022)](https://openreview.net/forum?id=cGDAkQo1C0p) to account for distributional shifts in the data.

### Location-specific deep learning
Going back to the data, I noticed that there was a lot of variability in the mean crop yield between locations. This variance is also not geographically driven, as it can be a political decision to support farming of certain crops in specific countries with sharp borders.
<img width="880" height="587" alt="image" src="https://github.com/user-attachments/assets/cde3f4b2-7c90-44c6-b600-2d7cc7b10fe1" />

The idea was then to simply do deep learning over the entire sequence of wheather data per crop per location, extracting conditional patterns to predict the yield of each year. 
This generalises slightly better than the mean per location per crop, but still suffers from overfitting by essentially performing regression onto the training data.



