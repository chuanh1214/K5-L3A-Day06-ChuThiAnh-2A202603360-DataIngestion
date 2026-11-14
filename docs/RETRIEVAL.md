# Chạy chunking, embedding và retrieval

Các lệnh dưới đây chạy từ thư mục `starter`, trong môi trường Python của dự án.
Database đã được người dùng khởi động bằng Docker Compose project `starter`; không cần tạo project thứ hai.

```powershell
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m scripts.validate_benchmark
python -m unittest discover -s tests -v
python -m src.run --stage dev
```

## Cấu hình cố định trước test

| Cấu hình | Chunking | Truy xuất |
|---|---|---|
| A | Cửa sổ 500 ký tự; overlap 50 | Không filter |
| B | Gom dòng/đoạn theo heading; tối đa 500 ký tự; không overlap | Không filter |
| C | Chính xác cùng chunk ID, text, metadata và vector như B | Filter chapter khi câu hỏi cung cấp |

Do text PDF hầu như không giữ dòng trống, B dùng heuristic heading/dòng; không tuyên bố khôi phục hoàn toàn đoạn văn hoặc thụt lề code.
Chunk luôn nằm trong một trang nguồn, giữ offset ký tự để truy vết. Các khoảng whitespace ở biên được bỏ, không bỏ ký tự nội dung.
ID chunk được băm từ nội dung, nguồn, trang, offset và metadata; chạy lại cùng đầu vào cho cùng ID.

Embedding local: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
Model đa ngữ dùng cùng `encode()` cho document/query, vector chuẩn hóa L2, không thêm prefix; CPU, seed 0, bốn thread.
Theo [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), vector có 384 chiều và model có giới hạn 128 token.
Pipeline đo lại giới hạn thực tế của tokenizer, tính cả special token. Chunk vượt giới hạn được tách tiếp bằng cùng token guard cho A/B/C; query quá dài báo lỗi thay vì cắt ngầm.
Lần chạy đầu phải tải model thật; model/cache lưu trong `starter/models` và `starter/.cache`, không đưa vào Git.
Không có API trả phí trong pipeline; chi phí dịch vụ trả phí ghi 0 USD, không coi đây là chi phí điện/phần cứng bằng 0.

Vector store là PostgreSQL + pgvector thật; tìm kiếm cosine exact, không tạo approximate vector index cho corpus nhỏ này.
Filter JSONB được áp dụng trong SQL trước xếp hạng; mọi giá trị được truyền bằng SQL parameters.
B và C dùng chung kết quả embedding nhưng có bản ghi riêng theo strategy để kiểm chứng filter là khác biệt duy nhất.

## Cố định rồi chạy test

Dev tạo `results/retrieval/config_lock.json` và `dev_selection.json` trước khi test.
Ưu tiên dev Evidence Hit@3, sau đó Hit@1, sau đó ít chunk hơn; dùng tên strategy để phá hòa tuyệt đối.
Lock bao gồm hash PDF, manifest, text corpus, benchmark và mã nguồn pipeline.

```powershell
python -m src.run --stage test
```

Test chỉ đọc index đã nạp ở dev và không xây/chỉnh chunk theo câu test.
Nếu đầu vào hoặc code thay đổi sau dev, test dừng. Khi đã có test summary, không ghi đè test hoặc chạy dev vào cùng thư mục kết quả.
Để làm một thí nghiệm mới, dùng `--output results/retrieval_new` cho cả dev và test, ghi rõ đây là lần chạy mới; không coi test đã xem là holdout mới.

## Kết quả và giới hạn

- `chunks_A.jsonl`, `chunks_B.jsonl`, `chunks_C.jsonl`: toàn bộ text, ID, nguồn/trang, offset và metadata.
- `index_stats.json`: số chunk, phân bố độ dài, token, thời gian, kiểm tra nạp lại không trùng.
- `dev_environment.json`, `test_environment.json`: thời điểm, model/revision, số chiều, phiên bản thư viện/database và xác minh mở lại kết nối vẫn có dữ liệu.
- `dev_traces.jsonl`, `test_traces.jsonl`: top-3 đầy đủ, score cosine, metadata, gold chính/thay thế và filter thực sự áp dụng.
- `dev_summary.json`, `test_summary.json`: Hit@1/Hit@3 cả tử/mẫu và tỷ lệ; compliance filter, N/A khi không có câu filter.
- `pipeline.log`: lỗi thật nếu có; không âm thầm fallback sang vector giả hoặc backend khác.

Chỉ 52 trang được chấp nhận trong [manifest](../starter/starter/data/manifest.json) được đưa vào index.
`upsert()` thay nguyên snapshot của một collection/strategy trong transaction: không nhân đôi và xóa chunk cũ trong snapshot đó.
Collection có namespace theo hash cấu hình: dữ liệu/model thay đổi tạo collection mới; collection cũ vẫn được giữ để tái lập, không được truy xuất cho lần chạy mới.

Chỉ số tự động dùng đúng nguồn/trang và toàn bộ quote sau chuẩn hóa NFC/whitespace, chấp nhận `gold_alternatives`.
Không cộng bằng chứng nằm rải qua nhiều chunk thành hit chính. Bằng chứng tương đương chưa liệt kê có thể bị tính thiếu: cần audit thủ công trước khi kết luận.
Đã bổ sung [grounding audit](../starter/results/retrieval/grounding_audit.csv), [metadata audit](../starter/results/retrieval/metadata_audit.csv) và [coherence audit](../starter/results/retrieval/coherence_audit.csv). Đây là nhãn AI hỗ trợ có công bố phương pháp trong [báo cáo cá nhân](../report/individual/2A202603360_ChuThiAnh.md), không phải chấm độc lập không dùng AI. Grounding/metadata đã được tác giả xác nhận trong trao đổi; coherence mới bổ sung cần tác giả xem lại. Summary matcher tự động vẫn giữ nguyên; filter compliance tự động không thay đánh giá ngữ nghĩa.
