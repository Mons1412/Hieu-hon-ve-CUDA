# Báo cáo rà soát nhãn Retrieval — v0.8.3 (SƠ BỘ)

**Nguồn:** CSV do người dùng cung cấp `relevance_preliminary_v082.csv`, gồm 160 kết quả truy xuất (20 câu × 2 collection × 4 chunks).

**Phạm vi đánh giá:** Các đoạn `evidence` tối đa 800 ký tự trong CSV; **không** xác minh trực tiếp với toàn văn PDF. Các điểm bên dưới là **đề xuất của AI** để người dùng duyệt; không phải nhãn vàng độc lập hay kết quả benchmark chính thức.

## Tình trạng dữ liệu

- 160/160 kết quả có **nhãn đề xuất** 0/1/2.
- 16/160 kết quả đã có **nhãn người dùng/nhãn trước đó**, được bảo toàn, không ghi đè.
- 144/160 dòng ban đầu không có điểm.
- 79/160 đoạn chứa đúng 800 ký tự, **có khả năng bị cắt**, cần ưu tiên kiểm tra toàn văn.
- 1/16 điểm ban đầu khác với đề xuất: `SM01 / BASELINE / Rank 4`, người dùng `0` và đề xuất `1`. Đây là bất đồng về rubric, **không phải lỗi của người dùng**.
- 4 câu `OOS01–OOS04` được chấm 0 cho cả 32 chunks do không có bằng chứng liên quan đến câu hỏi ngoại phạm vi trong các đoạn trích.

## Quy tắc

- **2:** Đoạn trích đủ hỗ trợ câu trả lời trọng tâm.
- **1:** Có liên quan, song chỉ hỗ trợ một phần hoặc thiếu cơ chế cần hỏi.
- **0:** Không hỗ trợ câu hỏi cụ thể.
- Không sử dụng cosine distance để suy ra nhãn; không dùng những gợi ý này làm ground truth trước khi rà soát độc lập.

## Chỉ số minh họa CHỈ từ nhãn đề xuất (16 câu CUDA, loại trừ 4 câu OOS)

| Chỉ số | Baseline | Section-Aware |
|---|---:|---:|
| Hit@4 | 16/16 (100.0%) | 16/16 (100.0%) |
| Precision@4 (micro; 64 chunks/collection) | 42.2% | 48.4% |
| MRR@4 | 0.786 | 0.823 |

**Cảnh báo:** Các con số trên là tự đánh giá bởi cùng mô hình gợi ý, có nguy cơ chủ quan và thiên lệch; không phải kết quả chính thức hoặc ước lượng độ chính xác trong sản phẩm. Cả hai collection dùng khoảng trang ≥7, cosine metric như log người dùng; phạm vi kiến thức và segmentation vẫn khác nhau.

## Theo câu hỏi — Top 4 (trong nhãn đề xuất)

| ID | Nhóm | Hit@4 (B/S) | Precision@4 (B/S) | RR@4 (B/S) |
|---|---|---|---|---|
| SM01 | Shared Memory | 1/1 | 0.50/0.25 | 1.00/1.00 |
| SM02 | Shared Memory | 1/1 | 0.25/0.50 | 1.00/1.00 |
| SM03 | Shared Memory | 1/1 | 0.25/0.25 | 1.00/1.00 |
| SM04 | Shared Memory | 1/1 | 0.25/0.50 | 0.33/1.00 |
| SM05 | Shared Memory | 1/1 | 0.50/0.50 | 1.00/0.50 |
| GM01 | Global Memory | 1/1 | 0.25/0.25 | 0.50/1.00 |
| GM02 | Global Memory | 1/1 | 0.75/0.75 | 1.00/1.00 |
| GM03 | Global Memory | 1/1 | 0.75/0.50 | 1.00/0.50 |
| GM04 | Global Memory | 1/1 | 0.25/0.25 | 0.25/1.00 |
| BC01 | Bank Conflicts | 1/1 | 0.50/0.75 | 1.00/1.00 |
| BC02 | Bank Conflicts | 1/1 | 0.50/0.50 | 1.00/0.33 |
| BC03 | Bank Conflicts | 1/1 | 0.25/0.25 | 0.50/1.00 |
| MM01 | Matrix Multiplication | 1/1 | 0.75/1.00 | 1.00/1.00 |
| MM02 | Matrix Multiplication | 1/1 | 0.25/0.50 | 0.50/0.50 |
| MM03 | Matrix Multiplication | 1/1 | 0.25/0.50 | 0.50/1.00 |
| MM04 | Matrix Multiplication | 1/1 | 0.50/0.50 | 1.00/0.33 |

## Những trường hợp cần ưu tiên kiểm tra

1. **SM01 · BASELINE · Rank 4 · trang 55:** điểm gốc 0, đề xuất 1. Đoạn nêu tác dụng của Shared Memory nhưng không trực tiếp định nghĩa; cả 0 và 1 có thể có lý nếu rubric khác nhau. Cần chốt rubric trước khi duyệt.
2. **SM01 · BASELINE · Rank 1 · trang 50:** điểm gốc 2; giải thích vai trò hợp tác giữa threads nhưng thiếu định nghĩa đầy đủ bộ nhớ on-chip. Giữ điểm gốc; có thể kiểm tra thêm nếu muốn tiêu chí “định nghĩa” nghiêm ngặt.
3. **BC02 · BASELINE · Rank 3 · trang 59:** nhắc tới *register bank conflicts*, không nên cho điểm cao với câu hỏi Shared Memory bank conflicts.
4. **MM01 · SECTION_AWARE · Rank 1 · trang 55:** tổng hợp lợi ích Shared Memory, nhưng section vẫn thuộc C=AAᵀ; cần xem câu hỏi đang hỏi tối ưu nhân ma trận nói chung hay riêng C=AB.
5. **MM04 · SECTION_AWARE · Rank 1 · trang 49:** mô tả latency Shared Memory, không có evidence về tile B → đề xuất 0.
6. **OOS01–OOS04:** tất cả chunks được đề xuất 0; đây là **relevance retrieval**, không chứng minh hệ thống sinh câu trả lời sẽ `ABSTAIN`.

## Hướng sử dụng

- Mở CSV/XLSX để lọc `review_priority`, đặc biệt `CAO - BẤT ĐỒNG` và `CAO - ĐOẠN CẮT`.
- So sánh trực tiếp với tài liệu NVIDIA trước khi gán `final_relevance`.
- Khi chốt nhãn, chỉ tính metric từ `final_relevance`, không trộn với `suggested_relevance`.
- Nên bổ sung tập câu hỏi/nhãn độc lập nếu đưa chỉ số vào README, CV hoặc báo cáo khoa học.
