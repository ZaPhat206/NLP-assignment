# Phân loại cảm xúc review phim IMDB

Bài thực hành so sánh SimpleRNN và LSTM với Logistic Regression, SVM và mạng neural cơ bản MLP. Mỗi review được phân loại thành **tiêu cực (0)** hoặc **tích cực (1)**.

Toàn bộ code, kết quả, biểu đồ và nhận xét nằm trong [main.ipynb](main.ipynb). Notebook đã lưu output nên có thể mở để xem mà không cần huấn luyện lại.

## Các file

| File | Nội dung |
| --- | --- |
| `main.ipynb` | Xử lý dữ liệu, huấn luyện, đánh giá và phân tích lỗi |
| `requirements.txt` | Các thư viện cần cài |

## Cài đặt và chạy

Dùng Python 3.11 và Conda. Mở terminal tại thư mục bài làm, tạo môi trường nếu chưa có:

```powershell
conda create -n NLP python=3.11 -y
conda activate NLP
```

Nếu đã có môi trường `NLP`, chỉ cần chạy `conda activate NLP`.

Để dùng GPU NVIDIA, cài bản PyTorch có CUDA trước. Lấy lệnh từ [trang cài đặt PyTorch](https://pytorch.org/get-started/locally/), chọn Windows, Pip, Python và CUDA phù hợp với máy. Chạy lệnh bằng `python -m pip` trong môi trường vừa kích hoạt, rồi cài các thư viện còn lại:

```powershell
python -m pip install -r requirements.txt
```

Kiểm tra Python và GPU:

```powershell
python -c "import sys; import torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Đường dẫn Python cần trỏ tới môi trường `NLP`. Nếu CUDA trả về `False`, notebook sẽ dùng CPU.

Trong VS Code, cài extension Python và Jupyter, mở `main.ipynb`, rồi chọn **Select Kernel → Python Environments → NLP**. Chạy các ô từ trên xuống bằng **Run All**. Sau khi thay đổi thư viện hoặc muốn chạy lại toàn bộ thí nghiệm, chọn **Restart Kernel** trước.

Lần chạy đầu cần Internet để tải dữ liệu. Sau khi chạy xong, lưu notebook để giữ các bảng kết quả và biểu đồ. Nếu kết quả thay đổi, cập nhật phần nhận xét và bảng trong README theo lần chạy mới.

## Dữ liệu và cách chia tập

Dùng [IMDB trên Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb), gồm 50.000 review có nhãn, cân bằng giữa hai lớp. Không sử dụng phần dữ liệu không nhãn.

| Tập | Số review | Mục đích |
| --- | ---: | --- |
| Train | 20.000 | Học từ điển, TF-IDF và trọng số mô hình |
| Validation | 5.000 | Theo dõi quá trình học, chọn epoch tốt nhất |
| Test | 25.000 | Đánh giá cuối cùng trên cùng dữ liệu cho cả năm mô hình |

Validation được tách từ tập train gốc theo tỉ lệ 20%, giữ cân bằng nhãn và dùng seed 42. Tập test gốc được giữ nguyên. TF-IDF và từ điển chỉ được học từ train.

## Cấu hình thí nghiệm

- **Logistic Regression, SVM và MLP:** TF-IDF trên toàn bộ review, tối đa 50.000 đặc trưng unigram/bigram. SVM dùng `LinearSVC`; MLP có một lớp ẩn 64 nút, ReLU.
- **RNN và LSTM:** từ điển 10.000 mục, giữ tối đa 200 token cuối mỗi review, padding bên phải. Embedding 64 chiều, một lớp RNN/LSTM 64 nút, dropout 0,3.
- **Huấn luyện mạng neural:** batch size 32, Adam với learning rate 0,001, tối đa 15 epoch. Dừng sớm sau 3 epoch không cải thiện validation loss và khôi phục mô hình ở epoch tốt nhất. RNN/LSTM dùng gradient clipping ở mức 1,0.

Lần chạy được lưu sử dụng Windows, Python 3.11, PyTorch `2.14.1+cu130` và scikit-learn `1.9.1`. Máy có RTX 2050 4 GB VRAM, RAM 12 GB. Logistic Regression, SVM và MLP chạy CPU; RNN và LSTM chạy GPU.

## Kết quả đã lưu trong notebook

Các điểm Precision, Recall và F1 tính cho lớp tích cực. Thời gian chỉ tính phần fit, không gồm tải dữ liệu, tiền xử lý hay đánh giá.

| Mô hình | Accuracy | Precision | Recall | F1 | Thời gian fit (giây) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0,8960 | 0,8935 | 0,8992 | 0,8963 | 0,92 |
| SVM | 0,8970 | 0,8993 | 0,8942 | 0,8967 | 0,46 |
| MLP | 0,9045 | 0,9045 | 0,9044 | 0,9045 | 273,83 |
| RNN | 0,7913 | 0,8164 | 0,7515 | 0,7826 | 81,43 |
| LSTM | 0,8646 | 0,8801 | 0,8443 | 0,8618 | 114,43 |

MLP có accuracy cao nhất trong lần thử này, đạt 90,45%. SVM có kết quả gần MLP nhưng fit nhanh hơn nhiều. LSTM tốt hơn SimpleRNN, dù cả hai chưa vượt các mô hình dùng TF-IDF.

MLP, RNN và LSTM lần lượt chạy 4, 8 và 9 epoch; trọng số được giữ lại ở epoch 1, 5 và 6. Train loss tiếp tục giảm trong khi validation loss tăng ở các epoch sau, nên tăng số epoch không bảo đảm kết quả tốt hơn.

Cần tính đến khác biệt đầu vào: TF-IDF dùng cả review, còn RNN/LSTM chỉ nhận 200 token cuối. Thời gian cũng phụ thuộc CPU/GPU và tải máy. Đây là kết quả của một seed và một cấu hình; khi chạy lại, số liệu có thể thay đổi.
