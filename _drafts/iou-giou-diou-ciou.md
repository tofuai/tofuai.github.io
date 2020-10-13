---
layout: post
title: "Generalize-IoU, Distance-IoU, Complete-IoU: Cải tiến loss cho bài toán Bounding Box Regression"
description: "Intersection over Union (IoU) là một trong những metric phổ biến nhất được sử dụng trong việc đánh giá các mô hình Object Detection. Tuy nhiên, hàm mất mát sử dụng phương pháp IoU còn tồn tại rất nhiều vấn đề, không tốt cho quá trình huấn luyện. Chúng ta hãy cùng xem xét các biến thể cải tiến IoU sau đây: GIoU, DIoU và CIoU."
thumb_image: "iou-giou-diou-ciou/compare-regression-errors.png"
tags: [deep-learning, computer-vision]
---


## Nội dung chính

1. [Giới thiệu](#1-giới-thiệu)
2. [IoU](#2-iou)
3. [Generalize-IoU](#3-generalize-iou)
4. [Distance-IoU](#4-distance-iou)
5. [Complete-IoU](#5-complete-iou)
6. [Thực nghiệm](#6-thực-nghiệm)
7. [Kết luận](#7-kết-luận)

## 1. Giới thiệu
Đối với bài toán Object Detection, chúng ta cần dự đoán và đưa ra vị trí cụ thể của vật thể trong ảnh. Phương pháp phổ biến nhất được sử dụng là Bounding Box Regression, khoanh vùng vị trí vật thể trong một hình hộp 4 cạnh. Hình hộp này có thể được định vị theo: 
* Vị trí 2 góc (left, top, right, bottom).
* Vị trí 1 góc và kích thước hộp (left, top, width, height).
* Vị trí tâm và kích thước hộp (center x, center y, width, height)

{% include image.html path="iou-giou-diou-ciou/yolo-v2-bb.png" path-detail="iou-giou-diou-ciou/yolo-v2-bb.png" alt="YOLOv2 Bounding Box" caption="YOLO-v2 Bounding Box được định vị theo vị trí tâm và kích thước bounding box." source="https://arxiv.org/pdf/1612.08242.pdf" %}


{% katexmm %}
Một lựa chọn phổ biến cho hàm mất mát đối với bài toán Bounding Box Regression là sử dụng $l_{n}$ loss để đánh giá sai số giữa Bounding Box dự đoán và nhãn. Tuy nhiên, $l_{n}$ loss không có tính bất biến với các kích cỡ bounding box khác nhau (bounding box to sẽ có loss lớn hơn nhiều so với bounding box bé). Trong YOLO-v1, nhóm tác giả đã sử dụng căn bậc hai cho chiều dài và chiều rộng để giảm thiểu tình trạng này.
{% endkatexmm %}

{% include image.html path="iou-giou-diou-ciou/yolo-v1-loss.png" path-detail="iou-giou-diou-ciou/yolo-v1-loss.png" alt="YOLOv1 Loss" caption="Hàm mất mát của YOLO-v1." source="https://arxiv.org/pdf/1506.02640.pdf" %}

Vì vậy, chúng ta sẽ xem xét đến một số phương pháp sử dụng hàm mất mát khác không bị ảnh hưởng bởi kích cỡ bounding box trong bài này: IoU loss, GIoU loss, DIoU loss vaf CIoU loss.

## 2. IoU

Intersection over Union (IoU) là một trong những metric phổ biến nhất được sử dụng trong việc đánh giá các mô hình Object Detection.

<div class="formular">
{% katexmm %}
$IoU = \frac{\lvert B \cap B^{gt} \rvert }{\lvert B \cup B^{gt} \rvert}$
{% endkatexmm %}
</div>

<div class="formular">
{% katexmm %}
$IoU = \frac{\lvert B \cap B^{gt} \rvert }{\lvert B \cup B^{gt} \rvert}$
{% endkatexmm %}
</div>

<div class="formular">
{% katexmm %}
$\mathcal{L}_{IoU} = 1 - \frac{\lvert B \cap B^{gt} \rvert }{\lvert B \cup B^{gt} \rvert}$
{% endkatexmm %}
</div>

## 3. Generalize-IoU

<div class="formular">
{% katexmm %}
$\mathcal{L}_{GIoU} = 1 - IoU + \frac{\lvert C - B \cap B^{gt} \rvert }{\lvert C \rvert}$
{% endkatexmm %}
</div>

{% include image.html path="iou-giou-diou-ciou/giou-diou.png" path-detail="iou-giou-diou-ciou/giou-diou.png" alt="GIoU Loss vs. DIoU Loss" caption="GIoU Loss có giá trị tương đương với IoU loss trong các trường hợp này, trong khi DIoU vẫn có sự khác biêt. Màu xanh và đỏ thể hiện cho bounding box nhãn và bounding dự đoán." source="https://arxiv.org/pdf/1911.08287.pdf" %}


## 4. Distance-IoU

<div class="formular">
{% katexmm %}
$\mathcal{L}_{DIoU} = 1 - IoU + \frac{\mathcal{p}^2(b, b^{gt})}{c^2}$
{% endkatexmm %}
</div>

{% include image.html path="iou-giou-diou-ciou/diou-distance-center.png" path-detail="iou-giou-diou-ciou/diou-distance-center.png" alt="DIoU Loss" caption="DIoU Loss cho bài toán Bounding Box Regression, thu nhỏ khoảng cách giữa tâm của bounding box dự đoán và tâm của bounding box nhãn." source="https://arxiv.org/pdf/1911.08287.pdf" %}



## 5. Complete-IoU

<div class="formular">
{% katexmm %}
$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\mathcal{\rho}^2(b, b^{gt})}{c^2} + \mathcal{\alpha}\mathcal{v}$
{% endkatexmm %}
</div>

<div class="formular">
{% katexmm %}
$\mathcal{v} = \frac{4}{\pi}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h})^2$
{% endkatexmm %}
</div>

<div class="formular">
{% katexmm %}
$\mathcal{\alpha} = \frac{\mathcal{v}}{(1 - IoU) + \mathcal{v}}$
{% endkatexmm %}
</div>



## 6. Thực nghiệm
## 7. Kết luận
