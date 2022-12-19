# minitorch

[minitorch](https://github.com/minitorch/minitorch) is a DIY project that implements the core internal concepts underpinning deep learning systems from scratch in native Python. Shout out and thank you to Sasha Rush ([@srush_nlp](https://twitter.com/srush_nlp)) and all contributors of minitorch for creating great open-source learning projects like this one.

## Training classifiers 
Classifiers can be trained using the notebooks in `/notebooks` or through a Streamlit app (see below). The repository defines four example training datasets for training the classifiers. Examples outputs of the training process for each dataset are shown below.

### Simple dataset
<div id="simple-dataset-plot-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="notebooks/plots/losses/loss-simple.png" width="350"/>
      	    </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/roc/roc-simple.png" width="350"/>
            </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/predictions/predictions-simple.png" width="350"/>
            </td>
        </tr>
    </table>
</div>


### Diagonal dataset
<div id="diagonal-dataset-plot-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="notebooks/plots/losses/loss-diagonal.png" width="350"/>
      	    </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/roc/roc-diagonal.png" width="350"/>
            </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/predictions/predictions-diagonal.png" width="350"/>
            </td>
        </tr>
    </table>
</div>

### Split dataset
<div id="split-dataset-plot-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="notebooks/plots/losses/loss-split.png" width="350"/>
      	    </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/roc/roc-split.png" width="350"/>
            </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/predictions/predictions-split.png" width="350"/>
            </td>
        </tr>
    </table>
</div>


### XOR dataset
<div id="xor-dataset-plot-table">
    <table>
	    <tr>
    	    <td style="padding:10px">
        	    <img src="notebooks/plots/losses/loss-xor.png" width="350"/>
      	    </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/roc/roc-xor.png" width="350"/>
            </td>
            <td style="padding:10px">
            	<img src="notebooks/plots/predictions/predictions-xor.png" width="350"/>
            </td>
        </tr>
    </table>
</div>


## Training classifiers through the app
... Coming soon ...


