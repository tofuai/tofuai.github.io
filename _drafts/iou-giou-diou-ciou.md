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
Một lựa chọn phổ biến cho hàm mất mát đối với bài toán Bounding Box Regression là sử dụng $l_{n}$ loss để đánh giá sai số giữa Bounding Box dự đoán và nhãn. Tuy nhiên, $l_{n}$ loss không có tính bất biến với các kích cỡ bounding box khác nhau (bounding box to sẽ có $l_{n}$ loss lớn hơn nhiều so với bounding box bé). Trong YOLO-v1, nhóm tác giả đã sử dụng căn bậc hai cho chiều dài và chiều rộng để giảm thiểu tình trạng này.
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

{% katexmm %}
IoU được tính toán dựa trên tỷ lệ giữa phần diện tích trùng nhau chia cho phần diện tích hợp nhất của 2 bounding box. Do đó, IoU hoàn toàn bất biến đối với kích thước của vật thể khi so sánh. Ngoài ra, việc sử dụng IoU loss thay cho $l_{n}$ giúp mô hình đạt kết quả tối ưu trên metric IoU hơn.
{% endkatexmm %}

<div class="formular">
{% katexmm %}
$\mathcal{L}_{IoU} = 1 - IoU$
{% endkatexmm %}
</div>

Tuy nhiên, IoU loss chỉ hoạt động khi hai bounding box chồng lên nhau, và không làm gradient dịch chuyển trong trường hợp không chồng lên nhau. Để giải quyết vấn đề này, chúng ta đến với một phiên bản nâng cấp hơn, Generalize-IoU (GIoU).

## 3. Generalize-IoU

Công thức của GIoU được mô tả như sau:

<div class="formular">
{% katexmm %}
$\mathcal{L}_{GIoU} = 1 - IoU + \frac{\lvert C - B \cap B^{gt} \rvert }{\lvert C \rvert}$
{% endkatexmm %}
</div>

{% katexmm %}
Trong đó, $C$ là bounding box bé nhất bao quanh $B$ và $B^{gt}$. Nhờ việc tính toán thêm dựa trên phần diện tích bao quanh 2 bounding box, ta có thể tránh được hiện tượng gradient vanishing đối với trường hợp 2 bounding box không chồng lên nhau.
{% endkatexmm %}

Tuy nhiên, khi sử dụng GIoU, mô hình lại gặp phải vấn đề hội tụ chậm và có đánh giá về vị trí không hiệu quả trong các trường hợp sau:

{% include image.html path="iou-giou-diou-ciou/giou-diou.png" path-detail="iou-giou-diou-ciou/giou-diou.png" alt="GIoU Loss vs. DIoU Loss" caption="GIoU Loss có giá trị tương đương với IoU loss trong các trường hợp này, trong khi DIoU vẫn có sự khác biêt. Màu xanh và đỏ thể hiện cho bounding box nhãn và bounding dự đoán." source="https://arxiv.org/pdf/1911.08287.pdf" %}


## 4. Distance-IoU

Khác với GIoU, DIoU thêm phần tính toán dựa vào khoảng cách tâm thay vì diện tích:

<div class="formular">
{% katexmm %}
$\mathcal{L}_{DIoU} = 1 - IoU + \frac{\mathcal{p}^2(b, b^{gt})}{c^2}$
{% endkatexmm %}
</div>

{% katexmm %}
Trong đó $b$ và $b_{gt}$ là tâm của hai bounding box $B$ và $B_{gt}$, $\mathcal{p}(.)$ là khoảng cách Euclidean, và $c$ là độ dài đường chéo của bounding box $C$, bounding box bé nhất bao quanh $B$ và $B^{gt}$. 
{% endkatexmm %}

{% include image.html path="iou-giou-diou-ciou/diou-distance-center.png" path-detail="iou-giou-diou-ciou/diou-distance-center.png" alt="DIoU Loss" caption="DIoU Loss cho bài toán Bounding Box Regression, thu nhỏ khoảng cách giữa tâm của bounding box dự đoán và tâm của bounding box nhãn." source="https://arxiv.org/pdf/1911.08287.pdf" %}

Trong khi GIoU loss có xu hướng mở rộng diện tích của bounding box dự đoán để tạo ra vùng chồng nhau trước, rồi sau đó tối ưu dựa trên tham số IoU. DIoU loss lại trực tiếp thu nhỏ khoảng cách tâm giữa 2 bounding box, giúp cho việc hội tụ diễn ra nhanh hơn.

{% include image.html path="iou-giou-diou-ciou/giou-diou-converge.png" path-detail="iou-giou-diou-ciou/giou-diou-converge.png" alt="DIoU Loss" caption="Tiến trình bounding box regression tạo bởi GIoU loss (hàng đầu tiên) và DIoU loss (hàng thứ hai). Màu xanh và đen thể hiện cho box nhãn và box anchor. Màu xanh và đỏ thể hiện cho box dự đoán bởi GIoU và DIoU. GIoU loss có xu hướng tăng diện tích của box dự đoán cho đến khi chồng lên box nhãn. Trong khi DIoU trực tiếp thu nhỏ khoảng cách tâm của box dự đoán và box nhãn." source="https://arxiv.org/pdf/1911.08287.pdf" %}


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
