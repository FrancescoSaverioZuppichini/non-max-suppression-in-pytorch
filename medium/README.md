https://gist.github.com/ad8118ad6923d411a00bd0cbe208e241

# Non Max Suppression (NMS) in PyTorch

![alt](/images/header.png)

Today we'll see how to implement non max suppression in PyTorch

*Code is [here](https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch) and an interactive version of this article can be found [here](https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch/blob/main/README.ipynb)*

## Preambula

If you are doing computer vision (especially object detection), you know what *non max suppression (nms)* is. There are a lot of good articles online giving a proper overview. In a nutshell, *non max suppression* reduces the number of output bounding boxes using some heuristics, e.g. intersection over union (iou).

From [`PyTorch` doc](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) 

>NMS iteratively removes lower-scoring boxes which have an IoU greater than iou_threshold with another (higher-scoring) box.

[This is an amazing article](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) from [ Sambasivarao K](https://medium.com/@SambasivaraoK) that goes trough *nms*, giving you a very good idea of what it does.

Now assume we know what it does, let's see how it works.

## Example

Let's load up an image and create bounding boxes


https://gist.github.com/3499d5692f0f097853492435d63a18d2




    
![png](/images/output_4_0.png)
    



Let me create two `bboxes`, one for the `head` and one for the `mic` 


https://gist.github.com/2b15163b5f0ba1d4e0a4171e8001d6c6

We have the `bboxes` in `[0, 1]` range, this is not necessary but it's useful when you have multiple classes (we'll see later why).


https://gist.github.com/8935f2dd8afaffda7f952b5d6a5de869




    <matplotlib.image.AxesImage at 0x7f354efe96d0>




    
![png](/images/output_8_1.png)
    


Let's add more overlapping bboxes


https://gist.github.com/100e480efcd1085fe9587a4759d6a673




    <matplotlib.image.AxesImage at 0x7f354e79de50>




    
![png](/images/output_10_1.png)
    


Okay, messy enough. We now have six `bboxes`. Let's also define `scores`, this is usually output by the model.


https://gist.github.com/0379c76ee70dd9422c065252ee065fc8

and our `labels`, `0` for *head*, `1` for *mic*


https://gist.github.com/ce54793fb4d3d2958a8963403dcabcd1

as a final touch, let's permute the data


https://gist.github.com/5b8eb839b212d2ae64cb9376293450c1


https://gist.github.com/5ac235e95772029ee18e4bfdbe7f937b

Let's see them


https://gist.github.com/95e13312221a66c952e137ef3042bdbc




    <matplotlib.image.AxesImage at 0x7f357b9ff070>




    
![png](/images/output_19_1.png)
    


good!

## Implementation

So NMS works by iteratively removing low-score overlapping bounding boxes. So, the steps are the following

https://gist.github.com/93251a6625a1cf18ac052477f7adb564

Mimicking the torch implementation, our `nms` takes three parameters (actually copied and pasted from torch's doc):

- boxes (`Tensor[N, 4]`)) – boxes to perform NMS on. They are expected to be in (`x1, y1, x2, y2`) format with `0 <= x1 < x2 and 0 <= y1 < y2`.

- scores (`Tensor[N]`) – scores for each one of the boxes

- iou_threshold (`float`) – discards all overlapping boxes with `IoU > iou_threshold`

**we will return the indices of the non suppressed bounding boxes**


https://gist.github.com/9b1d25366fb9da82c8f86ada0add02dd

let's go line by line

https://gist.github.com/ab05d54ad94b38f2aac258ba15782fd4

we get the sorted indices based on `scores`


https://gist.github.com/b377804d078c9f9384884fbfe2572dbe

we are creating the `indices` we will use to iterate `bboxes`

https://gist.github.com/475e8d2ca02ebd9ce5e8c484c0734a42

`keep` is a vector used to know if a `bbox` should be kept or not, if `keep[i] == 1` then `bboxes[order[i]]` is not suppressed

https://gist.github.com/68104487d93fc15abc6bd53861916667
we iterate over all `bboxes`


https://gist.github.com/c80a5261b0f928b964f1b371c8033e71
if current `bbox` is not suppressed `keep[i] = 1`

https://gist.github.com/d4ae70c7f6adf4ab833e2428761c9cf3

get `bbox` using the sorted position

https://gist.github.com/17cdce6577141912a77a043bfab4bd6e

calculate `iou` between current `bbox` and all the other candidates. Notice two things

https://gist.github.com/eb72ab2599ca16ebf28fed57b964d090

This will set to zero all the suppressed `bboxes` (since `keep` will be equal to `0`)

https://gist.github.com/aed279727f34b6593007dfa2144a82d8

We need to compare with all the next `bboxes` that come after in the sorted order and **we need to skip the current one**, so this is why we have a `+ 1`

https://gist.github.com/f6574e6a9f5bae5c2dcb2ff77e102aa9

select the indices where `iou` is greater than `iou_threshold`

https://gist.github.com/5f46f546dd9b6c122e5f3f9992f6c6ce

Since we sliced `bboxes` before, `(bboxes ...)[i + 1:])`, we need to add the offset to those indices, so we add `+ i + 1`.

Finally, we return `return order[keep]`, to map back to the original `bboxes` indices (non-sorted)



Okay, let's try it!


https://gist.github.com/a1f84c9acff164c8e162b32a6164231e




    <matplotlib.image.AxesImage at 0x7f354e3ea340>




    
![png](/images/output_25_1.png)
    


Since we have multiple classes, we need to make our `nms` compute iou within the same class. There is a nice **trick**. 

Remember our `bboxes` are between `[0,1]`? Well, we can add the `labels` to them to push away `bboxes` with different classes.


https://gist.github.com/13955c093d0ec3e0da2116ca469f4494




    <matplotlib.image.AxesImage at 0x7f354e36c460>




    
![png](/images/output_27_1.png)
    


we can make create an interactive version, to play with different thresholds


https://gist.github.com/65c100a9b9bafe73f0ad28d71b2beecf

    Enabling notebook extension jupyter-js-widgets/extension...
          - Validating: [32mOK[0m



https://gist.github.com/7d09ce471d7211bdbb2534892a75400b

    0.1



    
![png](/images/output_30_1.png)
    



    interactive(children=(FloatSlider(value=0.1, description='threshold', max=0.30000000000000004, min=-0.1), Outp…





    <function __main__._interact(threshold: float)>



Let's check the official `torch` implementation


https://gist.github.com/bc79da19dea7198dce2c3015451e2ff9




    <matplotlib.image.AxesImage at 0x7f357c228580>




    
![png](/images/output_32_1.png)
    


Same result! Let's see performance


https://gist.github.com/78bd337ee55363b18ddd71827b35c96b

    534 µs ± 22.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)



https://gist.github.com/1548218875f58ae88c29e26fdc8ccbe7

    54.4 µs ± 3.29 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


our implementation is around 10 times slower, I think It makes sense considering we are not using a custom cpp kernel!

If you think I've missed something, feel free to open an issue on [GitHub](https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch) :)

In this article we have seen how to implement non-max suppression in PyTorch, I hope it's not scary anymore!

See you in the next one 🚀

Francesco
