# `Transformer Architecture for Translation`
In this repository, we are implementing <span style='color:blue;'>`transformer`</span> architecture, according to the work,
[<span style='color:yellow;'> Attention is All You Need </span>](https://arxiv.org/abs/1706.03762), `from scratch` using **<span style='color:green;'>pytorch**</span>
and <span style='color:green;'>**numpy**</span>.At the beggining, I should thank [<span style='color:yellow;'> Ajay Halthor </span>](https://github.com/ajhalthor)
for his wonderfull toturial on transformer architecture.This implementation is used for translation task from **English** to 
**Persian**.The dataset used for training the model is extracted from [<span style='color:yellow;'> persiannlp </span>](https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa).The road to implement this architecture was to implement the base components of a 
transformer such as attnetion layers, embedding layers and feedforward layers and eventually assembling all togeather to make
encoder, decoder and finally the main transformer module.

## Dataset
The dataset contains 1.62M rows which was impossible for me to trian it with the limited resources that I had.So extracted a portion of 
<span style='color:green;'>**200k**</span> from the dataset which you can see a head of it in the followig table:
| persian                                | english                           |
| -------------------------------------- | --------------------------------- |
| آن وقت قاضی چه کرد؟	                 | What did the justice do?          |
| زودتر برویم. من حاضرم.                 | I am ready, my son, said Mercedes.|


## Training
Obviously this model is very large and needs to be trained on very powerfull gpus.To train this model, I used the 
`google colab` service but as the free usuage of gpu is limited, I had to train the model for only one epoch, store its state in a 
checkpoint and wait for 24 houurs to access the gpu one more time.Doing so, I trained the model for <span style='color:green;'>**12**</span> epochs, which is defienetly not 
sufficient but got some results that ensures the model architecture is truely implemented.

## Results
In this section, we can see some results of the model after being trained on <span style='color:green;'>**200k**</span> samples
and for <span style='color:green;'>**12**</span> epochs.


#### sample 1

> `English` : <span style='color:red;'>**yes , i can .**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**چرا  من مي‌تونم .**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**چرا  من ميزتونم .**</span>

#### sample 2

> `English` : <span style='color:red;'>**i reckon**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**من می‌گم**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**من می‌کم**</span>

#### sample 3

> `English` : <span style='color:red;'>**mommy yeah .**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**مامان بله .**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**مامان الو .**</span>

#### sample 4

> `English` : <span style='color:red;'>**he won't survive.**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**اون زنده نمي مونه,**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**اون زنده نمي بود**</span>

#### sample 5

> `English` : <span style='color:red;'>**i've told you they've found the boat.**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**من به تو گفته‌ام که آن‌ها قایق کوچک را پیدا کرده‌اند.**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**من به تو گفت  ام به آن ها قایق ورچک ر  تیدا برد  اند.**</span>

#### sample 6

> `English` : <span style='color:red;'>**but just because of this darkness he felt that the one guiding clue in the darkness was his work**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**اما دقیقا به دلیل همین تاریکی حس می‌کرد که کارش ریسمانی است که او را از ظلمت بیرون می‌کشد.**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**اما دقتقا به الی  ب ان داریکی چر می‌کرد که کردش میسمانی ایت اه ان بو بن ب مت بیرون می‌تهدا**</span>

#### sample 7

> `English` : <span style='color:red;'>**not even noticing the yellow butterflies that were still accompanying her.**</span>
>
> `Persian Translation`: <span style='color:yellow;'>**او حتی متوجه پروانه‌های زردرنگی که هنوز هم در بالای سرش پرواز می‌کردند نشد.**</span>
>
> `Model Prediction` : <span style='color:yellow;'>**او حتی مووبه برو ن  های زردر گی به هندی نا در بهرای سرش لروان می کر ند هدد.**</span>


As you see, when the length of sentence is long, the translator is not operating properly.But you can obviously see the high correlation between the model prediction and the actual persian translation.The reason is the low number of epochs that I could train the model.
12 epochs is obviously not sufficient for capturing patterns between persian and english language on this very large model.As a further work, we can train the model for more number of epochs, if actually resource is available, so that it can operate well on long sentences.

