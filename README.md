```python
# %matplotlib notebook
```

# Non Max Suppression (NMS) in PyTorch

![alt](/images/images/header.png)

Today we'll see how to implement non max suppression in PyTorch

*Code is [here](/images/https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch) and an interactive version of this article can be found [here](/images/https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch/blob/main/README.ipynb)*

## Preambula

If you are doing computer vision (especially object detection), you know what *non max suppression (nms)* is. There are a lot of good articles online giving a proper overview. In a nutshell, *non max suppression* reduces the number of output bounding boxes using some heuristics, e.g. intersection over union (iou).

From [`PyTorch` doc](/images/https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) 

>NMS iteratively removes lower-scoring boxes which have an IoU greater than iou_threshold with another (higher-scoring) box.

[This is an amazing article](/images/https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) from [ Sambasivarao K](/images/https://medium.com/@SambasivaraoK) that goes trough *nms*, giving you a very good idea of what it does.

Now assume we know what it does, let's see how it works.

## Example

Let's load up an image and create bounding boxes


```python
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# credit https://i0.wp.com/craffic.co.in/wp-content/uploads/2021/02/ai-remastered-rick-astley-never-gonna-give-you-up.jpg?w=1600&ssl=1
img = Image.open("./examples/never-gonna-give-you-up.webp")
img
```




    
![png](/images/output_4_0.png)
    



Let me create two `bboxes`, one for the `head` and one for the `mic` 


```python
original_bboxes = torch.tensor([
    # head
    [ 565, 73, 862, 373],
    # mic
    [807, 309, 865, 434]
]).float()

w, h = img.size
# we need them in range [0, 1]
original_bboxes[...,0] /= h
original_bboxes[...,1] /= w
original_bboxes[...,2] /= h
original_bboxes[...,3] /= w
```

We have the `bboxes` in `[0, 1]` range, this is not necessary but it's useful when you have multiple classes (we'll see later why).


```python
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor
from typing import List

def plot_bboxes(img : Image.Image, bboxes: torch.Tensor, *args, **kwargs) -> plt.Figure:
    w, h = img.size
    # from [0, 1] to image size
    bboxes = bboxes.clone()
    bboxes[...,0] *= h
    bboxes[...,1] *= w
    bboxes[...,2] *= h
    bboxes[...,3] *= w
    fig = plt.figure()
    img_with_bboxes = draw_bounding_boxes((to_tensor(img) * 255).to(torch.uint8), bboxes, *args, **kwargs, width=4)
    return plt.imshow(img_with_bboxes.permute(1,2,0).numpy())

plot_bboxes(img, original_bboxes, labels=["head", "mic"])
```




    <matplotlib.image.AxesImage at 0x7f6e6c633640>




    
![png](/images/output_8_1.png)
    


Let's add more overlapping bboxes


```python
max_bboxes = 3
scaling = torch.tensor([1, .96, .97, 1.02])
shifting = torch.tensor([0, 0.001, 0.002, -0.002])

# broadcasting magic (2, 1, 4) * (1, 3, 1)
bboxes = (original_bboxes[:,None,:] * scaling[..., None] + shifting[..., None]).view(-1, 4)

plot_bboxes(img, bboxes, colors=[*["yellow"] * 4, *["blue"] * 4], labels=[*["head"] * 4, *["mic"] * 4])
```




    <matplotlib.image.AxesImage at 0x7f6e6c578070>




    
![png](/images/output_10_1.png)
    


Okay, messy enough. We now have six `bboxes`. Let's also define `scores`, this is usually output by the model.


```python
scores = torch.tensor([
    0.98, 0.85, 0.5, 0.2, # for head
    1, 0.92, 0.3, 0.1 # for mic
])
```

and our `labels`, `0` for *head*, `1` for *mic*


```python
labels = torch.tensor([0,0,0,0,1,1,1,1])
```

as a final touch, let's permute the data


```python
perm = torch.randperm(scores.shape[0])
```


```python
bboxes = bboxes[perm]
scores = scores[perm]
labels = labels[perm]
```

Let's see them


```python
plot_bboxes(img, bboxes, 
            colors=["yellow" if el.item() == 0 else "blue" for el in labels], 
            labels=["head" if el.item()  == 0 else "mic" for el in labels]
           )
```




    <matplotlib.image.AxesImage at 0x7f6e6c4deee0>




    
![png](/images/output_19_1.png)
    


good!

## Implementation

So NMS works by iteratively removing low-score overlapping bounding boxes. So, the steps are the following

```
bboxes are sorted by score in decreasing order
init a vector keep with ones
for i in len(bboxes):
    # was suppressed
    if keep[i] == 0:
        continue
    # compare with all the others
    for j in len(bbox):
        if keep[j]:
            if (iou(bboxes[i], bboxes[j]) > iou_threshold):
                keep[j] = 0
    
return keep
```

Mimicking the torch implementation, our `nms` takes three parameters (actually copied and pasted from torch's doc):

- boxes (`Tensor[N, 4]`)) – boxes to perform NMS on. They are expected to be in (`x1, y1, x2, y2`) format with `0 <= x1 < x2 and 0 <= y1 < y2`.

- scores (`Tensor[N]`) – scores for each one of the boxes

- iou_threshold (`float`) – discards all overlapping boxes with `IoU > iou_threshold`

**we will return the indices of the non suppressed bounding boxes**


```python
from torchvision.ops.boxes import box_iou

def nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]
```

let's go line by line

```python
order = scores.argsort()
```

we get the sorted indices based on `scores`


```python
indices = torch.arange(bboxes.shape[0])
```

we are creating the `indices` we will use to iterate `bboxes`

```python
keep = torch.ones_like(indices, dtype=torch.bool)
```

`keep` is a vector used to know if a `bbox` should be kept or not, if `keep[i] == 1` then `bboxes[order[i]]` is not suppressed

```python
for i in indices:
    ...
```
we iterate over all `bboxes`


```python
     if keep[i]:
```
if current `bbox` is not suppressed `keep[i] = 1`

```python
        bbox = bboxes[order[i]]
```

get `bbox` using the sorted position

```
            iou = box_iou(bbox[None,...], (bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
```

calculate `iou` between current `bbox` and all the other candidates. Notice two things

```python
            (bboxes[order[i + 1:]] ... )
```

This will set to zero all the suppressed `bboxes` (since `keep` will be equal to `0`)

```python
            (bboxes[...] * keep[...][...,None])
```

We need to compare with all the next `bboxes` after in the sorted order and **we need to skip the current one**, so this is why we have a `+ 1`

```python
        overlapped = torch.nonzero(iou > iou_threshold)
```

select the indices where `iou` is greater than `iou_threshold`

```python
        keep[overlapped + i + 1] = 0
```

Since we sliced `bboxes` before, `(bboxes ...)[i + 1:])`, we need to add the offset to those indices, so we add `+ i + 1`.

Finally, we return `return order[keep]`, to map back to the original `bboxes` indices (non-sorted)



Okay, let's try it!


```python
nms_indices = nms(bboxes, scores, .45)
plot_bboxes(img, 
            bboxes[nms_indices],
            colors=["yellow" if el.item() == 0 else "blue" for el in labels[nms_indices]], 
            labels=["head" if el.item()  == 0 else "mic" for el in labels[nms_indices]]
           )
```




    <matplotlib.image.AxesImage at 0x7f6e9a2b7550>




    
![png](/images/output_25_1.png)
    


Since we have multiple classes, we need to make our `nms` compute iou within the same class. There is a nice **trick**. 

Remember our `bboxes` are between `[0,1]`? Well, we can add the `labels` to them to push away `bboxes` with different classes.


```python
nms_indices = nms(bboxes + labels[..., None], scores, .45)
plot_bboxes(img, 
            bboxes[nms_indices],
            colors=["yellow" if el.item() == 0 else "blue" for el in labels[nms_indices]], 
            labels=["head" if el.item()  == 0 else "mic" for el in labels[nms_indices]]
           )
```




    <matplotlib.image.AxesImage at 0x7f6e9991caf0>




    
![png](/images/output_27_1.png)
    


we can make create an interactive version, to play with different thresholds


```python
!jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

    Enabling notebook extension jupyter-js-widgets/extension...
          - Validating: [32mOK[0m



```python
from ipywidgets import interact

def _interact(threshold: float):
    print(threshold)
    nms_indices = nms(bboxes + labels[..., None], scores, threshold)
    plot_bboxes(img, 
                bboxes[nms_indices],
                colors=["yellow" if el.item() == 0 else "blue" for el in labels[nms_indices]], 
                labels=["head" if el.item()  == 0 else "mic" for el in labels[nms_indices]]
               )
    
interact(_interact, threshold = .1)
```


    interactive(children=(FloatSlider(value=0.1, description='threshold', max=0.30000000000000004, min=-0.1), Outp…





    <function __main__._interact(threshold: float)>



Let's check the official `torch` implementation


```python
from torchvision.ops.boxes import nms as torch_nms
nms_indices = torch_nms(bboxes + labels[..., None], scores, .45)
plot_bboxes(img, 
            bboxes[nms_indices],
            colors=["yellow" if el.item() == 0 else "blue" for el in labels[nms_indices]], 
            labels=["head" if el.item()  == 0 else "mic" for el in labels[nms_indices]]
           )
```




    <matplotlib.image.AxesImage at 0x7f6e9a166df0>




    
![png](/images/output_32_1.png)
    


Same result! Let's see performance


```python
%%timeit
nms(bboxes + labels[..., None], scores, .45)
```

    586 µs ± 22.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)



```python
%%timeit
torch_nms(bboxes + labels[..., None], scores, .45)
```

    51.2 µs ± 1.5 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


our implementation is around 10 times slower, I think It makes sense considering we are not using a custom cpp kernel!

If you think I've missed something, feel free to open an issue on [GitHub](/images/https://github.com/FrancescoSaverioZuppichini/non-max-suppression-in-pytorch) :)

In this article we have seen how to implement non-max suppression in PyTorch, I hope it's not scary anymore!

See you in the next one 🚀

Francesco
