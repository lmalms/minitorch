# minitorch

[minitorch](https://github.com/minitorch/minitorch) is a DIY project that implements the core internal concepts underpinning deep learning systems from scratch in native Python. Shout out and thank you to Sasha Rush [(@srush_nlp)](https://twitter.com/srush_nlp) and all contributors of minitorch for creating great open-source learning resources like this one.

<br/>

## Training classifiers 
Classifiers can be trained using the notebooks in `/notebooks` or through a Streamlit app (see below). The repository defines four example training datasets for training the classifiers. Examples outputs of the training process for each dataset can be found below.

### Simple dataset
Training Loss      |ROC Curve          |Predictions
:-----------------:|:-----------------:|:-----------------:
![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/losses/loss-simple.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/roc/roc-simple.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/predictions/predictions-simple.png)


### Diagonal dataset
Training Loss      |ROC Curve          |Predictions
:-----------------:|:-----------------:|:-----------------:
![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/losses/loss-diagonal.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/roc/roc-diagonal.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/predictions/predictions-diagonal.png)

### Split dataset
Training Loss      |ROC Curve          |Predictions
:-----------------:|:-----------------:|:-----------------:
![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/losses/loss-split.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/roc/roc-split.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/predictions/predictions-split.png)


### XOR dataset
Training Loss      |ROC Curve          |Predictions
:-----------------:|:-----------------:|:-----------------:
![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/losses/loss-xor.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/roc/roc-xor.png) | ![](https://github.com/lmalms/minitorch/blob/readme/notebooks/plots/predictions/predictions-xor.png)

## Training classifiers through the app
... Coming soon ...


<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="notebooks/plots/losses/loss-simple.png" width="200"/>
      	    </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/roc/roc-simple.png" width="300"/>
            </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/predictions/predictions-simple.png" width="400"/>
            </td>
        </tr>
    </table>
</div>