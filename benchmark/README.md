# Benchmark Think Python

## Phạm vi và nguồn

- Corpus: Think Python, 2nd Edition, Version 2.4.1, Allen B. Downey.
- Nguồn: <https://greenteapress.com/thinkpython2/thinkpython2.pdf>.
- Giấy phép: CC BY-NC 3.0; giữ ghi công và thông tin giấy phép khi chia sẻ.
- [Manifest](../starter/starter/data/manifest.json) chốt 52 trang được chấp nhận.
- SHA256 PDF tại thời điểm xây benchmark: `6033000bac3f9268ad75ee041bf8e5d9a3bc8a33bb156399cbd014918cb59f94`.
- `gold_page` dùng số trang PDF vật lý từ 1, không dùng số in ở chân trang.

## Cách chia trước thử nghiệm

[questions.json](questions.json) có 20 câu: q01-q15 là dev; q16-q20 là test.
Đây là phép chia cố định được thiết kế trước khi chạy retrieval, không phải chia ngẫu nhiên.
Dev phủ chuỗi, đọc file, list và dictionary; test gồm đảo chuỗi, sao chép list và tuple/zip.
Các câu dev/test cùng một sách, cùng ngôn ngữ và có chủ đề gần nhau; riêng q10 và q17 cùng trang 114 nhưng hỏi hai hành vi khác nhau.
Do đó test chỉ là holdout theo câu hỏi, không phải holdout theo tài liệu hay chương độc lập.
Hiện đã có kết quả dev/test và [báo cáo](../report/individual/2A202603360_ChuThiAnh.md). C được chọn theo dev trước khi chạy test; bước hoàn thiện báo cáo không tinh chỉnh chiến lược dựa vào điểm test.

Giữ cùng câu hỏi, corpus, model, store và metric giữa A/B/C.
Dùng dev để chọn cấu hình; ưu tiên Evidence Hit@3, sau đó Hit@1, sau đó ít chunk hơn nếu bằng điểm.
Chốt cấu hình trước khi chạy test. Không dùng điểm test để sửa chunking/filter rồi báo như test độc lập.
Nếu sửa câu hỏi hoặc nhãn sau khi đã xem kết quả, ghi rõ thay đổi và giới hạn trong báo cáo.

## Metadata filter và gold

- q04: `chapter=8`; q10: `chapter=10`; q18 (test): `chapter=12`.
- Chương được người dùng nêu ngay trong câu hỏi, là input trước truy xuất; không suy ra từ gold.
- A/B bỏ qua filter; C chỉ áp dụng filter được cung cấp. Với 17 câu không có filter, C chạy không filter.
- Chỉ gửi `question` và filter hợp lệ vào retriever; không gửi đáp án, trang hoặc nguồn gold.
- `gold_alternatives` là danh sách bằng chứng thay thế gồm `gold_source_id`, `gold_page`, `evidence_quote`, cùng cấu trúc gold chính.
- q08 có bằng chứng ở trang 106 hoặc 107; q10 ở trang 114 hoặc 121. Khi chấm, chấp nhận một gold hợp lệ, không bắt buộc gold chính.
- Hit yêu cầu một chunk đúng nguồn/trang và chứa đủ evidence hoặc bằng chứng tương đương được người chấm xác nhận. Các gold đã liệt kê không khẳng định bao phủ mọi cách diễn đạt tương đương.
- Lưu trace, báo riêng dev/test và cả tử/mẫu. Không dùng Hit@3 để thay nhãn coherence hay grounding thủ công.

## Kiểm chứng và giới hạn

Bộ câu hỏi/đáp án do AI hỗ trợ soạn và đối chiếu ngữ nghĩa với text trích xuất; học viên cần kiểm tra lại trước khi chốt bài.
Evidence giữ nguyên hiện tượng dính từ, mất thụt lề và gạch nối cuối dòng từ corpus để có thể kiểm chứng.
Chỉ chuẩn hóa Unicode NFC và whitespace khi so khớp; không sửa hoặc suy diễn nội dung nguồn.
Đáp án q07 chỉ nói bỏ ký tự xuống dòng theo ví dụ nguồn; q17 không bổ sung khẳng định về shallow/deep copy khi quote không chứng minh.
Corpus tiếng Anh nhưng câu hỏi tiếng Việt: đây là benchmark truy xuất xuyên ngôn ngữ; cần giữ một embedding model hỗ trợ cả hai ngôn ngữ khi làm bước embedding.

Từ thư mục `starter`, trong môi trường Python của dự án, chạy:

```powershell
python -m scripts.validate_benchmark
```

[Script kiểm tra](../starter/scripts/validate_benchmark.py) chỉ đọc dữ liệu: kiểm tra schema, ID, split, trang được chấp nhận, quote trên đúng nguồn/trang, filter và SHA256 corpus.
Nó không tính điểm retrieval, không chứng minh ngữ nghĩa đáp án bằng phép so khớp chuỗi và không tạo nhãn đánh giá thủ công.

Chunking, embedding, vector indexing và retrieval đã chạy; cấu hình khóa và kết quả được lưu trong [thư mục kết quả](../starter/results/retrieval/). Các mục mô tả cách chia ở trên ghi lại thiết kế benchmark trước thử nghiệm, không phải trạng thái chưa chạy hiện tại.
