# KHẢO SÁT HIỆN TRẠNG VÀ XÁC ĐỊNH YÊU CẦU

## Phân tích hiện trạng

Trong bối cảnh thương mại điện tử phát triển mạnh mẽ, việc xây dựng một nền tảng bán hàng không chỉ dừng lại ở mô hình một cửa hàng mà đang chuyển dịch sang mô hình sàn giao dịch thương mại điện tử đa nhà cung cấp. Hiện tại, người tiêu dùng cần một nền tảng tích hợp nơi họ có thể tìm kiếm, lọc, sắp xếp sản phẩm từ nhiều nhà bán hàng khác nhau, quản lý giỏ hàng mượt mà, thanh toán trực tuyến an toàn và tương tác qua Chatbot AI. Về phía người bán, họ thiếu một công cụ tập trung để đăng bán sản phẩm, theo dõi đơn hàng, cập nhật kho và thống kê doanh thu qua biểu đồ trực quan. Về phía quản trị viên, cần có một hệ thống toàn diện để kiểm duyệt người bán, quản lý mã giảm giá, các chương trình khuyến mãi và tùy biến giao diện trang chủ động. Việc xây dựng một hệ thống E-commerce đa nhà cung cấp là giải pháp hoàn chỉnh, tự động hóa quy trình giao dịch, quản lý tài chính thông qua tích hợp cổng thanh toán và nâng cao trải nghiệm người dùng

## Phân tích yêu cầu

### Yêu cầu chức năng

#### Yêu cầu chức năng nghiệp vụ

> **Bộ phận: Khách vãng lai — Mã số: GUEST**

| STT | Nội dung (Công việc, Loại công việc, Quy định, Ghi chú) |
| --- | --- |
| 1 | Xem giao diện Trang chủ Tra cứu QĐ_KVL1 Hiển thị các Banner, Deals, Lưới danh mục. |
| 2 | Tìm kiếm, lọc sản phẩm Tra cứu QĐ_KVL2 Lọc theo giá, màu, thương hiệu, danh mục. |
| 3 | Xem chi tiết sản phẩm và Sản phẩm có liên quan Tra cứu QĐ_KVL3 Xem ảnh, giá, mô tả, tồn kho và các gợi ý sản phẩm cùng danh mục. |
| 4 | Xem các trang thông tin tĩnh (Xem bài viết, tin tức, FAQ, Chính sách giao hàng, Chính sách hoàn trả). Tra cứu QĐ_KVL4 Đọc tin tức từ hệ thống hoặc gian hàng. Xem FAQ, Chính sách giao hàng, Chính sách hoàn trả. |
| 5 | Xem đánh giá (Review) Tra cứu QĐ_KVL5 Chỉ xem, không được viết đánh giá. |
| 6 | Đăng ký / Đăng nhập Tương tác QĐ_KVL6 Yêu cầu bắt buộc để tiến hành mua hàng. |

*Bảng 1‑1: Bảng yêu cầu chức năng nghiệp vụ Khách vãng lai*

| STT | Mã số | Tên quy định & Mô tả chi tiết |
| --- | --- | --- |
| 1 | QĐ_KVL1 | Quy định hiển thị Trang chủ Dữ liệu trang chủ (Homepage data) được hệ thống public mở hoàn toàn, Khách vãng lai không cần truyền JWT Token vẫn có thể tải được danh sách Deals, Grid Category và các sản phẩm nổi bật. API public |
| 2 | QĐ_KVL2 | Quy định tìm kiếm và lọc Hỗ trợ tìm kiếm bằng từ khóa. Lọc nâng cao theo: Danh mục 3 cấp (Level 1, 2, 3), mức giá (Min/Max), % giảm giá tối thiểu, màu sắc và sắp xếp (Giá từ thấp đến cao/Cao xuống thấp). |
| 3 | QĐ_KVL3 | Quy định xem chi tiết sản phẩm Khách vãng lai được xem toàn bộ thông tin công khai của sản phẩm bao gồm: Giá niêm yết (MRP Price), Giá bán (Selling Price), % giảm giá, hình ảnh (từ Cloudinary), màu sắc và thông tin Cửa hàng (Seller). Ở cuối trang chi tiết, hệ thống tự động hiển thị danh sách các "Sản phẩm liên quan" (Related products) dựa trên cùng Category Level 3 để tăng trải nghiệm mua sắm. |
| 4 | QĐ_KVL4 | Quy định xem bài viết, tin tức Khách vãng lai có thể truy cập đọc các bài viết quảng bá, tin tức sự kiện do Admin phát hành trên toàn sàn hoặc bài viết nội bộ của từng gian hàng Seller, các trang thông tin tĩnh (Static pages) như FAQ, Điều khoản dịch vụ được public hoàn toàn để khách hàng tìm hiểu trước khi đăng ký. |
| 5 | QĐ_KVL5 | Quy định giới hạn đánh giá Khách vãng lai chỉ có quyền ĐỌC các số sao (Rating) và nội dung bình luận (Review) của các sản phẩm. Tuyệt đối không được phép GỬI đánh giá mới. |
| 6 | QĐ_KVL6 | Quy định giới hạn nghiệp vụ Mua sắm Khách vãng lai không có giỏ hàng (Cart) và không có Danh sách yêu thích (Wishlist). Nếu cố tình nhấn nút "Thêm vào giỏ" hoặc "Mua ngay", hệ thống (React Router) bắt buộc chuyển hướng (Redirect) sang trang Đăng nhập / Đăng ký qua OTP. |

*Bảng 1‑2: Bảng quy định/ công thức liên quan Khách vãng lai*

> **Bộ phận: Quản trị viên — Mã số: ADMIN**

| STT | Nội dung (Công việc, Loại công việc, Quy định, Ghi chú) |
| --- | --- |
| 1 | Quản lý kiểm duyệt Seller Tra cứu/Lưu trữ QĐ_AD1 Thay đổi trạng thái tài khoản |
| 2 | Quản lý danh mục sản phẩm Tra cứu/Lưu trữ QĐ_AD2 Quản lý cấu trúc 3 cấp |
| 3 | Quản lý trang chủ (Homepage) Lưu trữ/Cập nhật QĐ_AD3 Banner, lưới danh mục |
| 4 | Quản lý khuyến mãi (Deals & Coupons) Tra cứu/Lưu trữ QĐ_AD4 Áp dụng toàn sàn |
| 5 | Quản lý tài khoản toàn hệ thống Tra cứu/Lưu trữ QĐ_AD5 |
| 6 | Quản lý khiếu nại (Disputes) Tra cứu/Xử lý QĐ_AD6 Quyết định cuối cùng giữa Khách và Seller. |
| 7 | Xử lý Hoàn tiền Tính toán/Xử lý QĐ_AD7 |
| 8 | Cấu hình hệ thống Xu (Coins) Lưu trữ/Cập nhật QĐ_AD8 Thiết lập tỉ lệ quy đổi và hạn mức sử dụng Xu. |
| 9 | Đối soát doanh thu Seller Tính toán/Kết xuất CT_AD1 Quản lý dòng tiền và tính toán phí nền tảng. |
| 10 | Quản lý đơn vị vận chuyển Tra cứu/Lưu trữ QĐ_AD10 |
| 11 | Kiểm duyệt Sản phẩm Tra cứu/Xử lý QĐ_AD11 Phê duyệt (Approve) hoặc Từ chối (Reject) sản phẩm mới do Seller đăng tải. |
| 12 | Quản lý Cấu hình Phí nền tảng Cập nhật QĐ_AD12 Giao diện cấu hình % Platform fee thu từ Seller. |

*Bảng 1‑3: Bảng yêu cầu chức năng nghiệp vụ Admin*

| STT | Mã số | Tên quy định & Mô tả chi tiết |
| --- | --- | --- |
| 1 | QĐ_AD1 | Quy định kiểm duyệt Seller Khi Seller đăng ký (Cung cấp GST, thông tin ngân hàng), tài khoản ở trạng thái PENDING. Admin kiểm duyệt và chuyển thành ACTIVE. Nếu vi phạm, Admin có thể SUSPEND hoặc BAN. |
| 2 | QĐ_AD2 | Quy định quản lý danh mục Danh mục được tổ chức theo 3 cấp độ (Level 1, Level 2, Level 3). |
| 3 | QĐ_AD3 | Quy định quản lý trang chủ Admin được phép thiết lập lưới danh mục hiển thị, danh mục đồ điện tử, nội thất và cập nhật các hình ảnh hiển thị trên trang chủ. |
| 4 | QĐ_AD4 | Quy định quản lý khuyến mãi Admin tạo Deal giảm giá cho các danh mục hoặc phát hành Coupon chung (yêu cầu mã code, phần trăm giảm, thời hạn, giá trị tối thiểu). |
| 5 | QĐ_AD5 | Quy định quản lý tài khoản Quản lý quyền truy cập và thông tin của tất cả Customer và Seller. |
| 6 | QĐ_AD6 | Quy định xử lý khiếu nại Đối với các đơn hàng DISPUTED, Admin đóng vai trò trọng tài xem xét bằng chứng của cả Khách hàng và Seller. Phán quyết của Admin (Chấp nhận hoàn tiền hoặc Không chấp nhận) là quyết định cuối cùng. |
| 7 | QĐ_AD7 | Quy định hoàn tiền Khi yêu cầu hoàn tiền được duyệt (bởi Seller hoặc Admin), hệ thống Back-end tự động gọi API Refund của cổng thanh toán để trả tiền về tài khoản ngân hàng gốc của khách hàng. Trạng thái đơn chuyển thành REFUNDED |
| 8 | QĐ_AD8 | Quy định cấu hình Xu Admin thiết lập 3 thông số chính: 1. Tỉ lệ kiếm xu (VD: 1000đ = 1 Xu). 2. Tỉ giá tiêu xu (VD: 1 Xu = 1đ). 3. Hạn mức thanh toán (VD: Xu chỉ thanh toán tối đa 50% giá trị đơn hàng). |
| 9 | CT_AD1 | Công thức đối soát doanh thu Seller Doanh thu Seller thực nhận = Tổng Selling Price - Phí nền tảng (Áp dụng theo % được cấu hình tại QĐ_AD12). Lưu ý: Phần tiền mà khách hàng đã dùng Xu để trừ thẳng vào đơn hàng sẽ do Admin bù lại vào ví của Seller trong quá trình đối soát để không làm thiệt hại đến doanh thu của Seller. Yêu cầu đối soát dòng tiền minh bạch. |
| 10 | QĐ_AD10 | Quy định Quản lý Đơn vị Vận chuyển Admin có quyền quản lý danh sách các đối tác vận chuyển trên sàn. Cho phép Thêm mới, Cập nhật thông tin, hoặc Bật/Tắt (Active/Deactive) các ĐVVC. Chỉ các ĐVVC ở trạng thái Active mới được phép hiển thị ở bước xử lý đơn hàng của Seller và Khách hàng. |
| 11 | QĐ_AD11 | Quy định kiểm duyệt sản phẩm Để bảo vệ nền tảng khỏi hàng giả/hàng cấm, các sản phẩm do Seller đăng tải sẽ ở trạng thái chờ duyệt. Admin xem xét thông tin và Approve (cho phép hiển thị) hoặc Reject (yêu cầu sửa đổi). |
| 12 | QĐ_AD12 | Quy định thiết lập phí sàn Admin có quyền thiết lập tỷ lệ Phần trăm Phí nền tảng (Platform fee) áp dụng cho các giao dịch thành công thông qua giao diện cấu hình động, không fix cứng trong mã nguồn. |

*Bảng 1‑4: Bảng quy định/ công thức liên quan Admin*

> **Bộ phận: Nhà cung cấp (Seller) — Mã số: SL**

| STT | Nội dung (Công việc, Loại công việc, Quy định, Ghi chú) |
| --- | --- |
| 1 | Cập nhật hồ sơ & thanh toán Lưu trữ QĐ_SL1 Cung cấp GST, Ngân hàng |
| 2 | Quản lý sản phẩm Lưu trữ/Cập nhật QĐ_SL2 Thêm biến thể, giá bán |
| 3 | Xử lý đơn hàng Cập nhật QĐ_SL3 Tự động hóa qua API GHTK/Grab. |
| 4 | In phiếu giao hàng (Vận đơn) Kết xuất QĐ_SL4 Mã vạch/QR code để dán lên gói hàng. |
| 5 | Báo cáo doanh thu & Thống kê Kết xuất CT_SL1 Xem biểu đồ doanh số |
| 6 | Xử lý Yêu cầu trả hàng/Hoàn tiền Tra cứu/Xử lý QĐ_SL5 Phê duyệt hoặc từ chối yêu cầu từ khách. |
| 7 | Cập nhật thống kê hoàn tiền Kết xuất CT_SL2 Tự động cập nhật vào Total Refund |
| 8 | Chat trực tuyến (Real-time) Tương tác QĐ_SL6 Nhắn tin hỗ trợ trực tiếp Khách hàng. |
| 9 | Xuất báo cáo Excel Kết xuất QĐ_SL7 Kết xuất dữ liệu đối soát ra file .xlsx. |

*Bảng 1‑5: Bảng yêu cầu chức năng nghiệp vụ Seller*

| STT | Mã số | Tên quy định & Mô tả chi tiết |
| --- | --- | --- |
| 1 | QĐ_SL1 | Quy định cập nhật hồ sơ Seller phải cung cấp mã số thuế doanh nghiệp (GST), địa chỉ kho hàng (Pickup address), và thông tin ngân hàng hợp lệ để đối soát. |
| 2 | QĐ_SL2 | Quy định đăng tải sản phẩm Mỗi sản phẩm phải thuộc 1 danh mục Level 3. Bắt buộc có Giá gốc (MRP) và Giá bán thực tế (Selling Price). Hệ thống tự động tính % Discount. Hình ảnh lưu qua Cloudinary. |
| 3 | QĐ_SL3 | Quy định xử lý đơn hàng & Vận chuyển Khi đơn hàng có trạng thái CONFIRMED, Seller sử dụng chức năng "Đẩy đơn vận chuyển". Hệ thống Back-end tự động gọi API của ĐVVC (GHTK/Grab) để lấy Mã vận đơn. Các trạng thái tiếp theo (SHIPPED, DELIVERED) sẽ do Webhook của ĐVVC tự động cập nhật về hệ thống, Seller không cần thao tác tay. Đồng bộ Real- time |
| 4 | CT_SL1 | Tính toán báo cáo doanh thu Báo cáo tự động tổng hợp: Tổng thu nhập (Total Earning), Tổng số đơn (Total Orders), Đơn bị hủy (Canceled Orders) và thể hiện qua biểu đồ trực quan. |
| 5 | QĐ_SL4 | Quy định in vận đơn Sau khi đẩy đơn thành công, Seller được phép kết xuất và in Phiếu giao hàng chứa Mã vận đơn (Tracking Code) định dạng PDF để đóng gói. |
| 6 | QĐ_SL5 | Quy định xử lý yêu cầu trả hàng Xử lý yêu cầu trả hàng | Khi nhận được yêu cầu RETURN_REQUESTED, Seller có tối đa 3 ngày để phản hồi. - Nếu Chấp nhận: Chờ nhận lại hàng, sau đó xác nhận để hệ thống hoàn tiền. - Nếu Từ chối: Phải ghi rõ lý do từ chối. |
| 7 | CT_SL2 | Cập nhật tổng hoàn tiền (Total Refund). Khi một đơn hàng hoàn tiền thành công, số tiền này bị trừ khỏi tổng thu nhập (Total Earning) và được cộng dồn vào thống kê Total Refund trên Seller Dashboard |
| 8 | QĐ_SL6 | Quy định hệ thống chat trực tuyến Hệ thống cung cấp công cụ Chat Real-time (sử dụng WebSockets) giúp Seller giải đáp thắc mắc của khách hàng về sản phẩm/đơn hàng ngay lập tức để tăng tỷ lệ chốt Sale. |
| 9 | QĐ_SL7 | Quy định xuất báo cáo Cho phép Seller trích xuất toàn bộ dữ liệu Lịch sử giao dịch (Transactions) và Doanh thu ra tệp tin Excel (.xlsx) để phục vụ nghiệp vụ đối soát và kế toán nội bộ. |

*Bảng 1‑6: Bảng quy định/ công thức liên quan Seller*

> **Bộ phận: Khách hàng (Customer) — Mã số: KH**

| STT | Nội dung (Công việc, Loại công việc, Quy định, Ghi chú) |
| --- | --- |
| 1 | Xác thực tài khoản bằng OTP Tương tác/Xác thực QĐ_KH1 |
| 2 | Tương tác Chatbot AI Tra cứu QĐ_KH2 Hỏi đáp tự động. |
| 3 | Tìm kiếm, lọc sản phẩm Tra cứu QĐ_KH3 Lọc theo giá, màu, danh mục. |
| 4 | Quản lý giỏ hàng & Wishlist Tương tác/Lưu trữ QĐ_KH4 |
| 5 | Đặt hàng & thanh toán Xử lý/Tính toán QĐ_KH5 Thanh toán qua Stripe/Razorpay. |
| 6 | Theo dõi & quản lý đơn hàng Tra cứu QĐ_KH6 Xem trạng thái đơn hàng. |
| 7 | Đánh giá & Phản hồi Kết xuất QĐ_KH7 Rating 1-5 sao, kèm ảnh thực tế. |
| 8 | Nhận thông báo, ưu đãi, hỗ trợ Tra cứu/Tương tác QĐ_KH8 Hỗ trợ qua chat, email hoặc hotline. |
| 9 | Yêu cầu trả hàng & hoàn tiền Tương tác/Cập nhật QĐ_KH9 Chỉ áp dụng cho đơn đã giao thành công. |
| 10 | Khiếu nại lên Admin (Dispute) Tương tác/Xử lý QĐ_KH10 Dùng khi Seller từ chối yêu cầu trả hàng. |
| 11 | Quản lý ví Xu (Reward Coins) Tra cứu QĐ_KH11 Xem số dư xu hiện tại và lịch sử nhận/tiêu xu. |
| 12 | Áp dụng Xu khi thanh toán Tính toán/Xử lý QĐ_KH12 Trừ tiền tương ứng với số xu khách muốn sử dụng. |
| 13 | Chat trực tuyến với Người bán (Real-time Tương tác QĐ_KH13 Nhắn tin trực tiếp theo thời gian thực (WebSockets) với Seller để nhận tư vấn cụ thể về món hàng. |
| 14 | Quản lý đa địa chỉ giao hàng Cập nhật/Lưu trữ QĐ_KH14 Thêm mới, cập nhật, xóa các địa chỉ trong Sổ địa chỉ. |

*Bảng 1‑7: Bảng yêu cầu chức năng nghiệp vụ Khách hàng*

| STT | Mã số | Tên quy định & Mô tả chi tiết |
| --- | --- | --- |
| 1 | QĐ_KH1 | Quy định đăng nhập bằng OTP Xác thực thông qua email sử dụng Java Mail Sender. Hệ thống gửi OTP gồm 6 chữ số để đăng ký/đăng nhập, thời gian hiệu lực giới hạn. |
| 2 | QĐ_KH2 | Quy định tương tác Chatbot AI Chatbot có khả năng truy xuất cơ sở dữ liệu để trả lời các câu hỏi về: Tình trạng đơn hàng, tổng tiền giỏ hàng, thông tin chi tiết sản phẩm và khuyến mãi. |
| 3 | QĐ_KH3 | Quy định tìm kiếm và lọc Hỗ trợ tìm kiếm theo từ khóa. Lọc nâng cao theo: Danh mục, mức giá (Min/Max), % giảm giá tối thiểu, màu sắc và sắp xếp (Giá từ thấp đến cao/Cao xuống thấp). |
| 4 | QĐ_KH4 | Quy định tách đơn hàng giỏ hàng Nghiệp vụ cốt lõi: Một giỏ hàng có thể chứa sản phẩm từ nhiều Seller. Khi Checkout, hệ thống nhóm các món hàng theo Seller ID thành các Orders riêng biệt tương ứng với từng Seller, nhưng gộp chung vào |
| 1 | PaymentOrder duy nhất để thanh toán 1 lần. Tính toán tổng tiền: CT_KH1 |  |
| 5 | QĐ_KH5 | Quy định thanh toán quốc tế/nội địa Hỗ trợ cổng thanh toán VnPay, SePay hoặc Momo. Thanh toán thành công sẽ đổi trạng thái PaymentOrder thành SUCCESS và tự động trừ hàng trong kho. |
| 6 | QĐ_KH6 | Quy định theo dõi & quản lý đơn hàng Khách hàng tra cứu nhật ký vận chuyển trực tiếp ngay trên giao diện website thông qua thanh tiến trình (Order Stepper). Giao tiếp qua API ĐVVC Hệ thống liên tục đồng bộ và hiển thị chi tiết các mốc thời gian, vị trí và trạng thái giao hàng từ Đơn vị vận chuyển (GHTK/Grab). Khách hàng được cung cấp Mã vận đơn để đối chiếu nếu cần, nhưng không bắt buộc phải rời khỏi sàn để tra cứu. Lịch sử đơn hàng được lưu trữ tối thiểu 12 tháng. |
| 7 | QĐ_KH7 | Quy định đánh giá (Review) Chỉ được đánh giá sản phẩm sau khi đã nhận hàng (DELIVERED). Chấm điểm từ 1 đến 5 sao, kèm bình luận và cho phép đính kèm hình ảnh thực tế sản phẩm. |
| 8 | QĐ_KH8 | Quy định nhận thông báo, ưu đãi, hỗ trợ Hệ thống gửi thông báo về đơn hàng, khuyến mãi, sự kiện. Hỗ trợ khách hàng qua chat, email hoặc hotline. |
| 9 | QĐ_KH9 | Quy định yêu cầu trả hàng/hoàn tiền Khách hàng chỉ được gửi yêu cầu trả hàng/hoàn tiền đối với đơn hàng có trạng thái là DELIVERED (Đã giao) trong vòng 7 ngày kể từ ngày nhận. Bắt buộc phải cung cấp lý do (hàng lỗi, sai mẫu...) và đính kèm hình ảnh/video minh chứng. Trạng thái đơn hàng chuyển sang RETURN_REQUESTED. |
| 10 | QĐ_KH10 | Quy định khiếu nại (Escalate) Nếu Người bán từ chối yêu cầu trả hàng, Khách hàng có quyền nhấn nút "Khiếu nại lên Admin". Đơn hàng chuyển sang trạng thái DISPUTED (Đang tranh chấp) để Admin can thiệp. |
| 11 | QĐ_KH11 | Quy định Tích xu Tích điểm (hoặc xu) cho khách hàng dựa trên lịch sử mua hàng. Khi đơn hàng đạt trạng thái DELIVERED, hệ thống tự động cộng số Xu = Final Payment Amount * Tỉ lệ tích xu (VD: 1%). |
| 12 | QĐ_KH12 | Quy định Tiêu xu Khách hàng có thể dùng Xu ở bước Thanh toán. Số tiền giảm được trừ trực tiếp vào Final Payment Amount. Có thể kết hợp sử dụng Xu và Mã giảm giá (Coupon) cùng lúc. |
| 13 | CT_KH1 | Công thức tính tiền Giỏ hàng Tổng giá bán (Total Selling Price) = ∑ (Giá bán × Số lượng) Giảm giá Coupon = Total Selling Price × (% Coupon / 100) Công thức tổng quát áp dụng tại bước Checkout, tự động đối Giảm giá Xu = Số Xu sử dụng × Tỉ giá Phí vận chuyển (Dynamic Shipping Fee): Được Back-end gọi API bên thứ 3 tính toán tự động dựa trên khoảng cách địa chỉ Kho Seller và Khách nhận. Tổng thanh toán (Final Payment) = Total Selling Price - Coupon - Xu + Phí vận chuyển. soát cả Mã giảm giá và Xu thưởng. |
| 14 | QĐ_KH13 | Quy định Chat trực tuyến với Seller Khách hàng được cung cấp khung Chat để trò chuyện trực tiếp với Người bán (Seller) của sản phẩm đó. Tin nhắn phải được cập nhật ngay lập tức (Real-time) ở cả 2 phía mà không cần tải lại trang Yêu cầu tích hợp WebSockets. |
| 15 | QĐ_KH14 | Quy định quản lý địa chỉ giao hàng Khách hàng có quyền tạo và quản lý nhiều địa chỉ giao hàng khác nhau (Nhà riêng, Cơ quan) trong Hồ sơ cá nhân. Tại bước Thanh toán (Checkout), khách hàng có thể chọn nhanh địa chỉ đã lưu hoặc tạo địa chỉ mới. |

*Bảng 1‑8: Bảng quy định/ công thức liên quan Khách hàng*

#### Yêu cầu chức năng hệ thống

• Môi trường: Hệ thống được xây dựng trên nền tảng Web Application, hoạt động qua mạng Internet, hỗ trợ truy cập trên máy tính và thiết bị di động. Front-end phát triển bằng React, TypeScript, Tailwind CSS, MUI và Redux Toolkit. Back-end sử dụng Java Spring Boot, MySQL Database. Hệ thống tích hợp trực tiếp với Cloudinary để lưu trữ phương tiện và các cổng thanh toán VnPay, SePay hoặc Momo. • Phân quyền: Hệ thống phân chia người dùng thành 3 nhóm quyền chính bằng Spring Security và JSON Web Token (JWT): o Khách hàng (ROLE_CUSTOMER): Có quyền truy cập giao diện cửa hàng, tra cứu sản phẩm, tương tác Chatbot, quản lý giỏ hàng, theo dõi đơn hàng cá nhân, và để lại đánh giá. Bị chặn truy cập vào các API thuộc quyền quản lý. o Người bán (ROLE_SELLER): Được cấp quyền vào bảng điều khiển (Seller Dashboard). Quản lý không gian bán hàng, đăng sản phẩm, theo dõi và xử lý các đơn hàng thuộc quyền sở hữu của mình, xem thống kê dòng tiền. o Quản trị viên (ROLE_ADMIN): Nắm quyền cao nhất vào bảng điều khiển hệ thống (Admin Dashboard). Quản lý trạng thái mọi tài khoản Seller/Customer, thiết lập giao diện Home page, cấu hình Coupon/Deal toàn sàn. Bảng Yêu cầu Hệ thống STT Nội dung Mô tả chi tiết Ghi chú 1 Nền tảng hoạt động Ứng dụng Web nhiều lớp (Client- Server 3-tier) xây dựng với React và Spring Boot. Hệ thống hoạt động qua mạng Internet, cho phép truy cập thông qua trình duyệt web trên máy tính hoặc thiết bị di động. Đảm bảo tương thích Responsive tốt trên đa thiết bị nhờ Tailwind CSS và MUI. 2 Tích hợp bên thứ ba Tích hợp VNPay, một số tài khoản ngân hàng; Java Mail Sender gửi mã Yêu cầu API Keys bảo mật chặt chẽ. OTP; Cloudinary lưu trữ tài nguyên hình ảnh. 3 Bảo mật và Phân quyền Quản lý luồng truy cập qua JWT Token. Mật khẩu người dùng băm qua BCrypt. Phân định rõ 3 roles: Admin, Seller, Customer. Ngăn chặn truy cập chéo giữa các Roles.

### Yêu cầu phi chức năng

#### Liên quan đến người dùng (Khách vãng lai, Khách hàng, Người bán, Quản trị viên)

trị viên): 
- **Tính tiến hóa (Khả năng mở rộng):
  - Hệ thống phải có kiến trúc linh hoạt, cho phép Quản trị viên (Admin) dễ dàng tùy biến giao diện trang chủ, cấu hình lưới danh mục sản phẩm, banner quảng cáo và các chương trình khuyến mãi (Deals) để phù hợp với thị hiếu thay đổi của khách hàng,.
  - Phải dễ dàng nâng cấp, mở rộng trong tương lai để đáp ứng quy mô số lượng lớn Người bán (Seller) tham gia vào sàn. Đồng thời, hệ thống được thiết kế sẵn sàng tích hợp thêm các dịch vụ giao hàng nội địa của bên thứ ba (như GHTK, GHN...) hoặc mở rộng các cổng thanh toán điện tử quốc tế mới bên cạnh Stripe và Razorpay hiện tại,. 
- **Tính tiện dụng (Dễ sử dụng - UX/UI):
  - Giao diện của hệ thống phải trực quan, thân thiện và có thiết kế đáp ứng (Responsive) hoàn toàn nhờ sử dụng Tailwind CSS và Material UI (MUI), đảm bảo hiển thị và thao tác mượt mà trên nhiều loại thiết bị (máy tính, máy tính bảng, điện thoại di động),,.
  - Các luồng thao tác cốt lõi của Khách hàng như: tìm kiếm, lọc sản phẩm đa tiêu chí (khoảng giá, màu sắc, % giảm giá), chọn biến thể kích cỡ, thao tác giỏ hàng và thanh toán gộp cho đa nhà cung cấp phải diễn ra đơn giản và tiện lợi nhất,,.
  - Hệ thống Bảng điều khiển (Dashboard) dành riêng cho Người bán và Quản trị viên phải được sắp xếp logic, các công cụ thống kê doanh thu thể hiện qua biểu đồ trực quan giúp giảm thiểu số lần nhấp chuột không cần thiết và nâng cao hiệu suất làm việc,,. 
- **Tính hiệu quả (Hiệu suất và Độ ổn định):
  - Nền tảng phải tối ưu hóa tốc độ tải trang (dưới 2 giây), phản hồi các thao tác tìm kiếm, chuyển đổi tab và phân trang (Pagination) nhanh chóng. Việc xử lý tải và lưu trữ khối lượng lớn hình ảnh sản phẩm đa phương tiện phải được thực hiện hiệu quả thông qua dịch vụ đám mây Cloudinary,.
  - Hệ thống phải hoạt động ổn định và xử lý chính xác dòng tiền đối soát, áp dụng đúng mã giảm giá (Coupon)/Xu thưởng cho hàng loạt giao dịch thanh toán trực tuyến diễn ra cùng lúc, đặc biệt trong các khung giờ cao điểm có lưu lượng truy cập khổng lồ,. 
- **Tính tương thích:
  - Hệ thống web hoạt động và hiển thị nhất quán trên các trình duyệt hiện đại phổ biến (Google Chrome, Safari, Firefox, Edge).
  - Phải đảm bảo tính đồng bộ dữ liệu theo thời gian thực (Real-time) giữa hoạt động đặt hàng của Khách hàng, tình trạng kho hàng và thông báo trạng thái vận chuyển hiển thị trên Bảng điều khiển của Người bán (Seller Dashboard)

#### Liên quan đến chuyên viên tin học (Đội ngũ phát triển)


- **Tính tái sử dụng:
  - Ứng dụng Front-end (React) cần được xây dựng theo kiến trúc Component (ví dụ: tái sử dụng các component ProductCard, OrderTable, AddressForm, DrawerList ở nhiều màn hình khác nhau),.
  - Cấu trúc Back-end (Spring Boot RESTful APIs) cần được tách biệt độc lập,, tuân thủ các chuẩn lập trình API giúp hệ thống dễ dàng được tái sử dụng để giao tiếp khi phát triển thêm nền tảng Ứng dụng di động (Mobile App) sau này. 
- **Tính bảo trì:
  - Mã nguồn dự án và cơ sở dữ liệu MySQL phải được phân tách theo các miền nghiệp vụ rõ ràng (Module Sản phẩm, Module Đơn hàng, Module Thanh toán, Module Tài khoản...),,. Cấu trúc này cho phép đội ngũ bảo trì dễ dàng sửa lỗi hoặc mở rộng tính năng mới ở một phân hệ mà không gây đổ vỡ (Crash) tới các module khác của hệ thống. 
- **Tính bảo mật:
  - Mật khẩu của người dùng bắt buộc phải được mã hóa một chiều an toàn bằng thuật toán BCrypt trước khi lưu trữ vào Cơ sở dữ liệu,.
  - Hệ thống phải áp dụng cơ chế xác thực phiên làm việc chặt chẽ bằng chuẩn JSON Web Token (JWT) thông qua bộ lọc Spring Security,,.
  - Cần thiết lập kiểm soát truy cập dựa trên vai trò (Role-Based Access Control) để ngăn chặn tuyệt đối tình trạng truy cập chéo tài nguyên giữa 3 nhóm quyền biệt lập: Khách hàng (ROLE_CUSTOMER), Người bán (ROLE_SELLER) và Quản trị viên (ROLE_ADMIN),,. Thông tin nhạy cảm về thẻ tín dụng khi thanh toán phải được tuân thủ chuẩn bảo mật trực tiếp thông qua API của VnPay, SePay hoặc Momo.

### Quy trình tác nghiệp

#### Quy trình tham quan và chuyển đổi của Khách vãng lai

Quy trình trải nghiệm và chuyển đổi của một Khách vãng lai trên sàn E-commerce diễn ra theo trình tự sau: Đầu tiên, Khách vãng lai truy cập vào nền tảng thông qua các trình duyệt web (có thể từ link chia sẻ, tìm kiếm Google hoặc trực tiếp URL). Hệ thống lập tức hiển thị Trang chủ (Homepage) với các Banner quảng cáo, các chương trình Khuyến mãi (Deals) đang diễn ra và các Danh mục nổi bật (Điện tử, Nội thất, Thời trang...) mà không yêu cầu đăng nhập. Tiếp theo, khách tự do điều hướng, sử dụng thanh tìm kiếm (Search) hoặc nhấp vào cây danh mục 3 cấp để duyệt sản phẩm. Tại trang danh sách, khách sử dụng bộ lọc nâng cao (lọc theo khoảng giá, màu sắc, thương hiệu) để thu hẹp kết quả. Khi tìm thấy sản phẩm ưng ý, khách nhấp vào để xem Chi tiết sản phẩm (đọc mô tả, xem hình ảnh thực tế, xem đánh giá 1-5 sao từ người dùng trước) hoặc đọc các bài viết/tin tức liên quan đến gian hàng đó để tăng độ tin cậy. Khi khách quyết định mua hàng và thực hiện hành động nhấn nút "Thêm vào giỏ hàng" (Add to Cart), "Thêm vào Wishlist", hoặc "Chat với AI Chatbot", hệ thống Spring Security ở Backend và React Router ở Frontend sẽ chặn thao tác này và tự động bật Pop-up / chuyển hướng (Redirect) khách sang Màn hình Đăng nhập / Đăng ký. Tại đây, khách vãng lai bắt buộc phải nhập Email/SĐT và xác thực mã OTP. Sau khi nhập OTP thành công, hệ thống cấp JWT Token, Khách vãng lai chính thức chuyển đổi trạng thái thành Khách hàng (Customer) và được tiếp tục quy trình mua sắm, thanh toán bị gián đoạn trước đó.

#### Quy trình khách hàng mua sắm trực tuyến

Quy trình khách hàng mua sắm diễn ra theo các bước sau: Đầu tiên, khách hàng truy cập vào nền tảng và tìm kiếm sản phẩm thông qua thanh tìm kiếm, bộ lọc nâng cao (theo mức giá, màu sắc, % giảm giá) hoặc nhận tư vấn trực tiếp từ AI Chatbot. Khi chọn được sản phẩm ưng ý, khách chọn biến thể (size, màu sắc) và thêm vào giỏ hàng hoặc đưa vào danh sách yêu thích (Wishlist) để mua sau. Tại bước thanh toán, do đặc thù đa nhà cung cấp, hệ thống sẽ tự động tách giỏ hàng thành các đơn hàng phụ tương ứng với từng người bán. Khách hàng tiến hành nhập mã giảm giá (nếu có) có thể tích chọn sử dụng "Xu tích lũy" từ ví tài khoản để trừ trực tiếp vào tổng số tiền phải trả. Hệ thống sẽ tự động tính toán lại số tiền cuối cùng. Thanh toán một lần duy nhất thông qua các cổng thanh toán trực tuyến an toàn như VnPay, SePay hoặc Momo. Cuối cùng, hệ thống ghi nhận giao dịch thành công và chuyển thông tin đơn hàng đến các gian hàng tương ứng.

#### Quy trình khách hàng tra cứu đơn hàng và tương tác

Để theo dõi đơn hàng, khách hàng đăng nhập vào hệ thống và chọn chức năng “Đơn hàng của tôi” (My Orders). Tại đây, họ có thể xem trạng thái hiện tại của đơn hàng, bao gồm các bước: Đã đặt (Placed), Đã xác nhận (Confirmed), Đang giao (Shipped) và Đã giao (Delivered). Điểm đặc biệt của hệ thống là khách hàng có thể mở giao diện AI Chatbot và hỏi trực tiếp bằng ngôn ngữ tự nhiên (ví dụ: "Tôi có bao nhiêu đơn hàng đã giao?") để tra cứu trạng thái đơn hàng hoặc chi tiết giỏ hàng nhanh chóng. Khi đơn hàng hoàn tất, hệ thống tự động kích hoạt tiến trình cộng Xu thưởng vào ví của khách hàng dựa trên tổng giá trị thanh toán của đơn hàng đó. Khách hàng có thể kiểm tra biến động số dư Xu tại màn hình Quản lý tài khoản cá nhân. Khách hàng có thể đánh giá (từ 1-5 sao) và đính kèm hình ảnh thực tế của sản phẩm. Nếu không vừa ý với món hàng, khách hàng có thể yêu cầu hoàn tiền, trả hàng với lý do hợp lý và tuân thủ đúng chính sách của sàn.

#### Quy trình quản lý gian hàng và sản phẩm (Dành cho Người bán)

Người bán (Seller) sau khi được cấp tài khoản sẽ đăng nhập vào hệ thống bảng điều khiển riêng (Seller Dashboard). Họ thực hiện các thao tác quản lý kho hàng bao gồm: thêm mới, chỉnh sửa hoặc xóa sản phẩm. Các thông tin cần cung cấp gồm có tên, mô tả, giá gốc (MRP), giá bán thực tế, số lượng tồn kho và tải hình ảnh lên hệ thống (thông qua Cloudinary). Khi có sự thay đổi về giá gốc và giá bán, hệ thống tự động tính toán phần trăm giảm giá để hiển thị. Ngoài ra, người bán cũng có trách nhiệm tiếp nhận đơn hàng từ khách và cập nhật trạng thái xử lý đơn (từ Chờ xử lý đến Đã giao hàng).

#### Quy trình kiểm duyệt và quản trị nền tảng (Dành cho Admin)

Quản trị viên (Admin) nắm quyền kiểm soát toàn bộ nền tảng thông qua Bảng điều khiển quản trị. Khi một người bán mới đăng ký, tài khoản sẽ ở trạng thái chờ duyệt (Pending Verification). Admin sẽ kiểm tra hồ sơ và thực hiện phê duyệt (Active), hoặc có thể đình chỉ (Suspend), cấm vĩnh viễn (Ban) đối với các tài khoản vi phạm chính sách. Bên cạnh quản lý người dùng, Admin thực hiện việc tùy chỉnh giao diện trang chủ, thay đổi lưới danh mục, banner, và phát hành các mã giảm giá (Coupon), chương trình khuyến mãi (Deals) cho toàn bộ hệ thống.

#### Quy trình thống kê và đối soát doanh thu

Hệ thống tự động tổng hợp và tính toán các chỉ số kinh doanh theo thời gian thực. Từ bảng điều khiển, Người bán có thể xem chi tiết tổng thu nhập, tổng số sản phẩm đã bán, số lượng đơn hàng bị hủy và theo dõi lịch sử dòng tiền (Transactions). Báo cáo doanh thu được xuất ra dưới dạng các biểu đồ trực quan (Earning graphs) theo ngày, tuần hoặc tháng, giúp người bán dễ dàng phân tích tình hình kinh doanh của gian hàng. Đồng thời, dữ liệu này là cơ sở để hệ thống tiến hành đối soát và thanh toán tiền hàng cho Người bán sau khi đơn hàng giao thành công.

#### Quy trình yêu cầu trả hàng và hoàn tiền (Refund & Return Process)

Quy trình xử lý trả hàng và hoàn tiền được diễn ra chặt chẽ giữa 3 bên nhằm đảm bảo tính công bằng: 
- Bước 1: Khởi tạo yêu cầu (Khách hàng): Khách hàng đăng nhập, truy cập lịch sử mua hàng và chọn đơn hàng có trạng thái "Đã giao" (Delivered) trong thời hạn cho phép (VD: 7 ngày). Khách hàng chọn chức năng "Yêu cầu trả hàng", điền lý do và tải lên hình ảnh/video minh chứng. Hệ thống chuyển trạng thái đơn sang "Yêu cầu trả hàng" (Return Requested) và tạm thời đóng băng khoản tiền đối soát của đơn hàng này đối với Người bán. 
- Bước 2: Xử lý yêu cầu (Người bán): Người bán nhận được thông báo trên Seller Dashboard. Xem xét minh chứng của khách hàng.
  - Trường hợp 2a (Đồng ý): Người bán bấm "Chấp nhận". Khách hàng gửi trả lại hàng. Khi Người bán nhận được hàng sẽ bấm "Xác nhận hoàn tiền". Trạng thái đơn chuyển thành "Đã hoàn tiền" (Refunded).
  - Trường hợp 2b (Từ chối): Người bán bấm "Từ chối" kèm theo lý do. 
- Bước 3: Khiếu nại (Khách hàng): Nếu bị Người bán từ chối, Khách hàng có quyền nhấn "Khiếu nại". Đơn hàng chuyển sang trạng thái "Tranh chấp" (Disputed). 
- Bước 4: Phán quyết (Quản trị viên - Admin): Quản trị viên can thiệp vào các đơn "Disputed", kiểm tra đối chứng dữ liệu từ cả hai bên. Admin đưa ra phán quyết cuối cùng. Nếu Admin duyệt hoàn tiền, hệ thống sẽ tự động kích hoạt API của cổng thanh toán để đẩy tiền về thẻ của khách, đồng thời hệ thống tự động cập nhật biểu đồ thống kê "Total Refund" (Tổng số tiền hoàn) trên Dashboard của Người bán

# MÔ HÌNH HÓA YÊU CẦU

## Nhận diện tác nhân và chức năng trong sơ đồ Use case

| Tác nhân (Actor) | Mã UC | Tên Use Case (User Goal) | Mô tả |
|---|---|---|---|
| Khách vãng lai (Guest) | UC01 | Khám phá nền tảng | Xem trang chủ (Banner, Deals), tìm kiếm, lọc sản phẩm nâng cao. Xem chi tiết sản phẩm, gợi ý sản phẩm liên quan và đọc các trang thông tin tĩnh (FAQ, Chính sách). |
| | UC02 | Đăng ký tài khoản | Khách vãng lai đăng ký tài khoản qua Email và xác thực bằng mã OTP để trở thành Khách hàng chính thức. |
| Khách hàng (Customer) | UC03 | Quản lý Giỏ hàng (Cart) | Thêm sản phẩm vào giỏ, cập nhật số lượng hoặc xóa sản phẩm. Tự động tính toán lại tổng tiền và % giảm giá. |
| | UC04 | Quản lý Danh sách yêu thích | Thêm các sản phẩm ưng ý vào Wishlist để lưu trữ cho các lần mua sắm sau và xóa sản phẩm khỏi danh sách. |
| | UC05 | Đặt hàng và Thanh toán | Chọn địa chỉ giao hàng, hệ thống tự động tách đơn theo Seller (Split Order), áp dụng Coupon/Xu, và thanh toán qua cổng điện tử. |
| | UC06 | Theo dõi & Quản lý đơn hàng | Xem lịch sử mua hàng, tra cứu tiến trình vận chuyển theo thời gian thực và hủy đơn khi còn ở trạng thái "Mới đặt". |
| | UC07 | Tương tác Chatbot AI | Nhắn tin hỏi đáp tự động bằng ngôn ngữ tự nhiên để tra cứu thông tin sản phẩm, đơn hàng, và giỏ hàng. |
| | UC08 | Đánh giá sản phẩm | Chấm điểm (1-5 sao), viết bình luận và đính kèm hình ảnh thực tế sau khi đơn hàng đã "Đã giao". |
| | UC09 | Yêu cầu Trả hàng & Hoàn tiền | Tạo yêu cầu đổi/trả hàng kèm minh chứng. Có quyền Khiếu nại (Dispute) lên Admin nếu bị Seller từ chối. |
| | UC10 | Quản lý Tài khoản & Ví Xu | Theo dõi biến động số dư Ví Xu, cập nhật thông tin cá nhân và quản lý Sổ đa địa chỉ giao hàng (Thêm/Sửa/Xóa). |
| Khách hàng & Người bán | UC11 | Chat trực tuyến (Real-time) | Luồng giao tiếp dùng chung kết nối người mua và người bán thông qua kiến trúc WebSockets để trao đổi, tư vấn trực tiếp về sản phẩm/đơn hàng. |
| Người bán (Seller) | UC12 | Quản lý Hồ sơ và Gian hàng | Cập nhật thông tin doanh nghiệp (GST), tài khoản ngân hàng đối soát, địa chỉ kho lấy hàng và trang trí Banner gian hàng. |
| | UC13 | Quản lý Kho sản phẩm | Đăng tải sản phẩm mới (chờ duyệt), tải ảnh qua Cloudinary, cập nhật giá bán/tồn kho và cấu hình màu sắc/kích cỡ. |
| | UC14 | Xử lý Đơn hàng & Vận chuyển | Tiếp nhận đơn, xác nhận, tự động đẩy đơn sang API hãng vận chuyển (GHTK/Grab) và kết xuất in Phiếu giao hàng. |
| | UC15 | Xử lý Yêu cầu Hoàn trả | Xem xét lý do và minh chứng từ Khách hàng để đưa ra quyết định Chấp nhận (cho phép hoàn tiền) hoặc Từ chối. |
| | UC16 | Theo dõi Đối soát & Doanh thu | Xem biểu đồ doanh thu tổng quan, theo dõi dòng tiền đối soát và thực hiện Xuất báo cáo dữ liệu ra file Excel. |
| Quản trị viên (Admin) | UC17 | Kiểm duyệt Người bán | Xét duyệt hồ sơ đăng ký gian hàng (Active), hoặc tạm đình chỉ (Suspend), cấm vĩnh viễn (Ban) tài khoản vi phạm. |
| | UC18 | Kiểm duyệt Sản phẩm | Xem xét thông tin các sản phẩm mới do Seller đăng tải để Phê duyệt (cho phép hiển thị công khai) hoặc Từ chối. |
| | UC19 | Quản lý Giao diện Trang chủ | Tùy biến linh hoạt giao diện trang chủ, cập nhật Banner, cấu hình lưới danh mục nổi bật mà không cần can thiệp code. |
| | UC20 | Quản lý Chiến dịch Khuyến mãi | Phát hành, cập nhật hoặc xóa các chiến dịch Marketing chung toàn sàn bao gồm Khuyến mãi (Deals) và Mã giảm giá (Coupons). |
| | UC21 | Giải quyết khiếu nại (Disputes) | Xem xét các đơn hàng có tranh chấp, đưa ra phán quyết cuối cùng và tự động gọi API hoàn tiền cho Khách hàng. |
| | UC22 | Quản lý Khách hàng | Xem danh sách khách hàng, theo dõi thông tin tài khoản và thực hiện khóa (Ban) các tài khoản vi phạm chính sách. |
| | UC23 | Cấu hình Thông số Tài chính | Thiết lập các thông số Ví Xu (tỉ lệ quy đổi, hạn mức tiêu) và cấu hình % Phí nền tảng (Platform fee) áp dụng cho gian hàng. |
| | UC24 | Theo dõi Nhật ký (Audit Log) | Truy xuất và xem xét lịch sử các thao tác thay đổi dữ liệu quan trọng trên hệ thống (kiểm duyệt, xóa dữ liệu) để kiểm toán. |
| Mọi Tác nhân | UC25 | Đăng nhập hệ thống | Người dùng xác thực danh tính qua Email/Mật khẩu (đối với Admin) hoặc đăng nhập không mật khẩu qua mã OTP (đối với Khách hàng, Người bán). Hệ thống cấp phiên làm việc và điều hướng theo phân quyền. |
| | UC26 | Đăng xuất | Người dùng chủ động kết thúc phiên làm việc. Hệ thống tiến hành hủy xóa khỏi Local Storage thiết bị và điều hướng người dùng về trang chủ mặt tiền một cách an toàn. |
| | UC27 | Đổi / Quên mật khẩu | Hỗ trợ người dùng yêu cầu thiết lập lại mật khẩu khi quên hoặc chủ động đổi mật khẩu để bảo vệ tài khoản. Quá trình này bắt buộc phải xác thực bảo mật thông qua mã OTP gửi về Email đã đăng ký. |

## Mô tả chi tiết từng tác nhân

| Tên tác nhân | Công việc/vai trò |
|---|---|
| Khách vãng lai (Guest) | Người dùng truy cập vào hệ thống nhưng chưa có tài khoản hoặc chưa đăng nhập. Họ có quyền tự do tham quan trang chủ, tìm kiếm sản phẩm, đọc chi tiết mô tả hàng hóa, đọc các bài viết tin tức và xem các chương trình khuyến mãi. Tuy nhiên, họ không được phép thao tác đặt hàng, thêm giỏ hàng hay đánh giá. Muốn thực hiện giao dịch, họ buộc phải đăng ký/đăng nhập. |
| Khách hàng (Customer) | Là Khách vãng lai đã thực hiện đăng ký và đăng nhập thành công. Họ có toàn quyền thực hiện các luồng mua sắm: thêm sản phẩm vào giỏ, tương tác Chatbot, đặt hàng, thanh toán trực tuyến, tích lũy/sử dụng xu thưởng và gửi yêu cầu hoàn tiền/đánh giá sau khi nhận hàng thành công. |
| Người bán (Seller) | Cá nhân/doanh nghiệp sở hữu gian hàng trên hệ thống. Đóng vai trò là nhà cung cấp hàng hóa, chịu trách nhiệm đăng tải sản phẩm, thiết kế giao diện gian hàng, xuất bản tin tức của shop và trực tiếp đóng gói, cập nhật trạng thái giao hàng cho khách. |
| Quản trị viên (Admin) | Người quản lý cấp cao nhất của hệ thống nền tảng. Đóng vai trò kiểm duyệt (tài khoản seller, luồng tiền hoàn trả, tranh chấp), duy trì cấu hình giao diện trang chủ, cấu hình hệ thống xu thưởng, và phát hành các chiến dịch Marketing chung toàn sàn (Coupons, Deals, Tin tức hệ thống). |

## Sơ đồ Use case

## Đặc tả Use case

### Use case 1

| Trường | Nội dung |
| --- | --- |
| Use Case ID | UC01 |
| Use Case Name | Khám phá nền tảng |
| Description | Là một Người dùng (Khách vãng lai hoặc Khách hàng), tôi muốn tham quan trang chủ, tìm kiếm, lọc và xem chi tiết sản phẩm để tìm được món hàng ưng ý. |
| Actor(s) | Khách vãng lai (Guest), Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Người dùng truy cập vào URL của hệ thống thông qua trình duyệt web. |
| Pre-Condition(s) | Thiết bị của người dùng có kết nối Internet. |
| Post-Condition(s) | Người dùng xem được các thông tin sản phẩm công khai trên nền tảng. |
| Basic Flow | 1. Người dùng truy cập vào nền tảng. 2. Hệ thống hiển thị Trang chủ với các chiến dịch Khuyến mãi (Deals), Banner quảng cáo và Lưới danh mục nổi bật. 3. Người dùng nhập từ khóa vào thanh tìm kiếm hoặc sử dụng bộ lọc nâng cao (khoảng giá, màu sắc, danh mục). 4. Hệ thống truy xuất dữ liệu và trả về danh sách các sản phẩm đáp ứng tiêu chí. 5. Người dùng nhấp chọn một sản phẩm cụ thể. 6. Hệ thống hiển thị màn hình Chi tiết sản phẩm bao gồm: hình ảnh, giá gốc, giá bán, mô tả, đánh giá (chỉ đọc) và danh sách các "Sản phẩm liên quan" ở cuối trang. |
| Alternative Flow | 3a. Người dùng nhấp vào các liên kết thông tin tĩnh. 3a1. Hệ thống điều hướng và hiển thị nội dung các trang: FAQ, Tin tức, Chính sách giao hàng, Chính sách hoàn trả. Use Case kết thúc. |
| Exception Flow | 4a. Hệ thống không tìm thấy sản phẩm nào khớp với từ khóa/bộ lọc. 4a1. Hệ thống hiển thị thông báo "Không tìm thấy sản phẩm phù hợp" và gợi ý xóa bộ lọc. Use Case quay lại bước 3. 6b. Khách vãng lai (Guest) cố tình nhấn nút "Thêm vào giỏ hàng" hoặc "Mua ngay". 6b1. Hệ thống chặn thao tác và hiển thị yêu cầu đăng nhập/đăng ký. Use Case kết thúc. |
| Business Rules | - BR01-1 (QĐ_KVL3): Mục "Sản phẩm liên quan" bắt buộc phải truy xuất các sản phẩm có cùng danh mục cấp 3 (Level 3 Category) với sản phẩm đang xem. - BR01-2 (QĐ_KVL5 & 6): Khách vãng lai tuyệt đối không có quyền tạo Giỏ hàng, Wishlist hay Gửi đánh giá. |
| Non-Functional Requirement | - NFR01-1: Tốc độ tải Trang chủ và Trang chi tiết sản phẩm phải dưới 2 giây dù có lưu lượng truy cập lớn. |

### Use case 2

| Trường | Nội dung |
| --- | --- |
| Use Case ID | UC02 |
| Use Case Name | Đăng ký tài khoản |
| Description | Là một Khách vãng lai, tôi muốn đăng ký tài khoản thành viên thông qua mã xác thực OTP để có thể thực hiện mua sắm trên sàn. |
| Actor(s) | Khách vãng lai (Guest) |
| Priority | Must Have |
| Trigger | Người dùng nhấn nút "Đăng ký" trên giao diện điều hướng. |
| Pre-Condition(s) | Người dùng chưa đăng nhập vào hệ thống. |
| Post-Condition(s) | Tài khoản được tạo thành công, người dùng được chuyển trạng thái thành Khách hàng (ROLE_CUSTOMER) và được cấp phiên làm việc. |
| Basic Flow | 1. Người dùng chọn "Đăng ký" trên màn hình. 2. Hệ thống hiển thị biểu mẫu yêu cầu cung cấp thông tin cơ bản (họ tên, ngày sinh, giới tính) và Email. 3. Người dùng nhập Email và nhấn "Gửi mã OTP". 4. Hệ thống kiểm tra tính hợp lệ và gửi mã OTP gồm 6 chữ số đến Email của người dùng. 5. Người dùng nhập mã OTP. 6. Người dùng chọn "Tạo tài khoản". 7. Hệ thống xác thực mã OTP. Nếu hợp lệ, hệ thống khởi tạo tài khoản Khách hàng mới. 8. Hệ thống thông báo đăng ký thành công, tự động đăng nhập và đưa người dùng về Trang chủ. |
| Alternative Flow | 3a. Người dùng chọn lệnh "Chuyển sang Đăng nhập". 3a1. Hệ thống chuyển đổi biểu mẫu sang màn hình Đăng nhập. Use Case chuyển tiếp sang Use Case Đăng nhập. |
| Exception Flow | 4a. Email người dùng cung cấp không đúng định dạng. 4a1. Hệ thống báo lỗi ngay tại ô nhập liệu và chặn lệnh gửi OTP. Use Case quay lại bước 3. 7b. Người dùng nhập mã OTP sai hoặc mã đã hết hạn hiệu lực. 7b1. Hệ thống hiển thị cảnh báo "Mã OTP không chính xác hoặc đã hết hạn". Use Case quay lại bước 5. |
| Business Rules | - BR02-1: Mã OTP chỉ bao gồm 6 chữ số ngẫu nhiên và có thời gian hiệu lực giới hạn. |
| Non-Functional Requirement | - NFR02-1: Hệ thống gửi Email chứa mã OTP trong thời gian không quá 5 giây kể từ lúc nhấn nút. |

### Use case 3

| Trường | Nội dung |
| --- | --- |
| Use Case ID | UC03 |
| Use Case Name | Quản lý Giỏ hàng (Cart) |
| Description | Là một Khách hàng, tôi muốn thêm sản phẩm, cập nhật số lượng hoặc xóa sản phẩm ra khỏi giỏ hàng của mình để chuẩn bị cho việc thanh toán. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng nhấn "Thêm vào giỏ" tại một sản phẩm hoặc truy cập trực tiếp vào màn hình Giỏ hàng. |
| Pre-Condition(s) | Khách hàng đã đăng nhập thành công. |
| Post-Condition(s) | Hệ thống ghi nhận đúng số lượng, tổng tiền và áp dụng mã giảm giá (nếu có) vào giỏ hàng. |
| Basic Flow | 1. Khách hàng lựa chọn một sản phẩm và chỉ định thao tác (Thêm mới, Tăng/Giảm số lượng, hoặc Xóa). 2. Hệ thống kiểm tra quyền hợp lệ và số lượng tồn kho của sản phẩm tương ứng. 3. Nếu hợp lệ, hệ thống thực thi việc cập nhật giỏ hàng theo yêu cầu. 4. Hệ thống tự động tính toán lại Tổng tiền gốc (Total MRP), Tổng tiền thanh toán (Total Selling Price) và Tổng số lượng món hàng. 5. Hệ thống hiển thị thông báo cập nhật thành công và hiển thị giao diện Giỏ hàng mới nhất. |
| Alternative Flow | 1a. Khách hàng nhập Mã giảm giá (Coupon Code) vào ô nhập liệu tại màn hình Giỏ hàng và nhấn "Áp dụng". 1a1. Hệ thống kiểm tra tính hợp lệ của mã giảm giá. 1a2. Hệ thống tính toán lại giá trị chiết khấu và trừ thẳng vào Tổng tiền thanh toán. Use Case quay lại bước 5. |
| Exception Flow | 2a. Khách hàng muốn tăng số lượng nhưng sản phẩm đó đã đạt mức tối đa tồn kho (Out of stock). 2a1. Hệ thống báo lỗi không thể thêm số lượng và chặn thao tác. Use Case dừng lại. 1a1.1 (Ngoại lệ của 1a): Mã giảm giá đã hết hạn, nhập sai, hoặc chưa đạt giá trị tối thiểu. 1a1.2. Hệ thống hiển thị thông báo "Mã giảm giá không hợp lệ hoặc đã được sử dụng". Use Case quay lại màn hình Giỏ hàng. |
| Business Rules | - BR03-1: Mỗi Khách hàng chỉ sở hữu duy nhất 1 Giỏ hàng (Mối quan hệ 1-1). - BR03-2 (CT_KH1): Tổng thanh toán Giỏ hàng = Tổng giá bán thực tế của các sản phẩm - Tiền giảm giá từ Coupon. |
| Non-Functional Requirement | - NFR03-1: Tính toán thay đổi tổng tiền phải diễn ra tức thì (Real-time logic) trên giao diện ngay khi khách hàng nhấn nút +/-. |

### Use case 4

| Trường | Nội dung |
| --- | --- |
| Use Case ID | UC04 |
| Use Case Name | Quản lý Danh sách yêu thích (Wishlist) |
| Description | Là một Khách hàng, tôi muốn lưu lại các sản phẩm mà mình quan tâm vào một danh sách riêng để dễ dàng theo dõi và đặt mua vào lần sau. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Should Have |
| Trigger | Khách hàng nhấn vào biểu tượng "Trái tim" trên thẻ sản phẩm hoặc truy cập menu Wishlist. |
| Pre-Condition(s) | Khách hàng đã đăng nhập thành công. |
| Post-Condition(s) | Sản phẩm được thêm vào hoặc loại bỏ khỏi Wishlist. |
| Basic Flow | 1. Khách hàng thao tác nhấn biểu tượng "Trái tim" trên một sản phẩm. 2. Hệ thống kiểm tra sản phẩm này đã tồn tại trong Danh sách yêu thích của khách hàng hay chưa. 3. Hệ thống ghi nhận sản phẩm vào Cơ sở dữ liệu Wishlist của khách hàng. 4. Hệ thống đổi màu biểu tượng "Trái tim" (highlight) và hiển thị thông báo đã thêm thành công. |
| Alternative Flow | 2a. Hệ thống phát hiện sản phẩm ĐÃ TỒN TẠI trong Danh sách yêu thích. 2a1. Hệ thống hiểu đây là lệnh Xóa (Remove). 2a2. Hệ thống gỡ sản phẩm ra khỏi Wishlist, bỏ highlight biểu tượng "Trái tim" và thông báo đã xóa thành công. Use Case kết thúc. 1b. Khách hàng truy cập trang quản lý Danh sách yêu thích. 1b1. Hệ thống truy xuất và hiển thị lưới toàn bộ sản phẩm khách hàng đã lưu. 1b2. Khách hàng nhấp vào icon "Dấu X" (Close) trên thẻ sản phẩm. Use Case tiếp tục thực hiện luồng 2a2. |
| Exception Flow | Xử lý lỗi rớt mạng (Timeout). |
| Business Rules | - BR04-1: Mỗi người dùng chỉ có đúng 1 Danh sách yêu thích (1-1), một sản phẩm chỉ được xuất hiện 1 lần trong danh sách đó. |
| Non-Functional Requirement | - NFR04-1: Trạng thái thêm/xóa khỏi Wishlist phải được lưu động bộ vào cơ sở dữ liệu và hiển thị phản hồi giao diện không cần tải lại trang. |

### Use case 5

| Trường | Nội dung |
| --- | --- |
| Use Case ID | UC05 |
| Use Case Name | Đặt hàng và Thanh toán |
| Description | Là một Khách hàng, tôi muốn chọn địa chỉ, phương thức thanh toán (trực tuyến hoặc COD) và hệ thống sẽ tự động đề xuất/áp dụng mã giảm giá tốt nhất cho tôi trước khi tôi trả tiền. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng nhấn nút "Checkout" (Tiến hành đặt hàng) tại màn hình Giỏ hàng. |
| Pre-Condition(s) | Khách hàng đã đăng nhập và Giỏ hàng đang chứa ít nhất 1 sản phẩm. |
| Post-Condition(s) | Đơn hàng được tạo thành công, hàng trong kho bị trừ, giao dịch thanh toán được ghi nhận (hoặc ghi nhận chờ thu hộ đối với COD). |
| Basic Flow | 1. Khách hàng bắt đầu tiến trình Checkout. 2. Hệ thống hiển thị màn hình Thanh toán gồm: Sổ địa chỉ, Phương thức thanh toán (VNPay/SePay/Momo/COD), Danh sách sản phẩm, và Khung tóm tắt đơn hàng. 3. Khách hàng lựa chọn 1 Địa chỉ giao nhận hiện có và chọn 1 Phương thức thanh toán. 4. [Voucher Engine]: Dựa trên Tổng tiền hàng, Địa chỉ giao nhận (tính Phí ship) và Phương thức thanh toán vừa chọn, hệ thống tự động quét và áp dụng Mã giảm giá (Coupon) mang lại mức chiết khấu cao nhất cho khách hàng. 5. (Tùy chọn) Khách hàng nhập yêu cầu áp dụng Xu tích lũy vào đơn hàng. 6. Hệ thống tính toán tổng thanh toán cuối cùng (Final Payment). 7. Khách hàng nhấn lệnh "Xác nhận và Thanh toán" (Pay Now). 8. Hệ thống tự động Tách giỏ hàng (Split Order): gom nhóm sản phẩm theo Seller thành các Đơn hàng phụ, nhưng gộp chung giá trị vào một lệnh thanh toán. 9. [Thanh toán trực tuyến]: Khách hàng hoàn tất nhập thông tin tại cổng thanh toán bảo mật của bên thứ 3. 10. Webhook trả kết quả về hệ thống. Đơn hàng chuyển trạng thái "Đã đặt" (PLACED), hiển thị màn hình "Giao dịch thành công". |
| Alternative Flow | 3a. Khách hàng chọn "Thêm địa chỉ mới". 3a1. Khách hàng điền biểu mẫu thông tin và lưu lại. Hệ thống lưu vào Sổ địa chỉ. Use Case quay lại bước 3 và tự động chạy lại bước 4. 4a. Khách hàng muốn đổi mã giảm giá khác: 4a1. Khách hàng nhấn vào mục "Chọn mã giảm giá". 4a2. Khách hàng chọn 1 mã khác từ Kho Voucher hoặc nhập mã thủ công và nhấn Áp dụng. Use Case đi tiếp đến bước 5. 9b. [Thanh toán COD]: Khách hàng đã chọn phương thức "Thanh toán khi nhận hàng" (COD) ở bước 3. 9b1. Hệ thống KHÔNG chuyển hướng sang cổng thanh toán mà tạo đơn hàng trực tiếp với trạng thái PLACED và PaymentOrder ở trạng thái COD_PENDING (Chờ thu hộ). 9b2. Hệ thống trừ hàng tồn kho, xóa giỏ hàng, gửi thông báo cho Seller. 9b3. Hiển thị màn hình "Đặt hàng thành công". Khi Shipper giao hàng và thu tiền mặt, trạng thái PaymentOrder sẽ được ĐVVC cập nhật qua Webhook sang COD_COLLECTED. Use Case kết thúc. |
| Exception Flow | 4a2.1 (Ngoại lệ của 4a): Khách hàng nhập thủ công một mã giảm giá đã hết hạn, hoặc không thỏa mãn điều kiện Phương thức thanh toán/Giá trị tối thiểu. 4a2.2: Hệ thống báo lỗi "Mã không hợp lệ hoặc chưa đủ điều kiện" và từ chối áp dụng. Use Case quay lại bước 4. 9a. Thanh toán trực tuyến thất bại hoặc khách hàng hủy giao dịch tại Cổng thanh toán. 9a1. Hệ thống ghi nhận thanh toán thất bại, không tạo đơn hàng thành công. Use Case dừng lại. |
| Business Rules | - BR05-1 (Thuật toán Auto-apply): Khi có sự thay đổi về Địa chỉ hoặc Phương thức thanh toán, Voucher Engine bắt buộc phải quét lại toàn bộ dữ liệu để cập nhật lại mã giảm giá và phí ship theo thời gian thực. - BR05-2 (QĐ_KH4): Bắt buộc tách Order riêng biệt theo mã Seller_ID. - BR05-3 (CT_KH1): Final Payment = (Tổng tiền hàng) - (Coupon) - (Quy đổi Xu) + (Phí vận chuyển). - BR05-4 (COD): Đơn hàng COD không được phép sử dụng Xu tích lũy (chỉ cho phép Coupon). Một số Coupon có thể giới hạn chỉ áp dụng cho thanh toán trực tuyến. Phí thu hộ COD (nếu có) được ĐVVC tính thêm vào Phí vận chuyển. |
| Non-Functional Requirement | - NFR05-1: Thuật toán quét và áp dụng Voucher phải được xử lý ở tốc độ cao (dưới 1s) để không gây giật lag giao diện (giữ UX mượt mà). - NFR05-2: Thông tin thẻ không lưu trên DB hệ thống, truyền mã hóa 100% qua cổng VnPay/SePay/Momo. |

### Use case 6

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC06 |
| Use Case Name | Theo dõi & Quản lý đơn hàng |
| Description | Là một Khách hàng, tôi muốn xem lại lịch sử các đơn đã đặt, tra cứu tiến trình giao hàng theo thời gian thực và hủy đơn nếu thay đổi ý định mua sắm. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng truy cập vào mục \"Đơn hàng của tôi\" (My Orders) trên giao diện. |
| Pre-Condition(s) | Khách hàng đã đăng nhập thành công vào hệ thống. |
| Post-Condition(s) | Khách hàng nắm bắt được trạng thái đơn hàng. Nếu thực hiện thao tác hủy, hệ thống ghi nhận trạng thái Hủy (Canceled) và hoàn lại kho. |
| Basic Flow | 1\. Khách hàng chọn mục \"Đơn hàng của tôi\". 2\. Hệ thống truy xuất và hiển thị danh sách toàn bộ đơn hàng (lịch sử mua sắm). 3\. Khách hàng nhấp vào một đơn hàng cụ thể để xem chi tiết. 4\. Hệ thống hiển thị màn hình Chi tiết Đơn hàng, bao gồm thanh tiến trình (Order Stepper) trực quan với các mốc: Đã đặt (Placed), Đã xác nhận (Confirmed), Đang giao (Shipped), và Đã giao (Delivered). 5\. Khách hàng theo dõi vị trí và tiến trình. Thao tác xem hoàn tất. |
| Alternative Flow | **3a. Khách hàng muốn hủy đơn hàng:** 3a1. Tại màn hình chi tiết, khách hàng chọn lệnh \"Hủy đơn hàng\" (Cancel Order). 3a2. Hệ thống yêu cầu xác nhận. Khách hàng đồng ý. 3a3. Hệ thống đổi trạng thái đơn sang \"Đã hủy\" (Canceled), tự động cộng lại số lượng sản phẩm vào kho tồn của Seller và thông báo hủy thành công. |
| Exception Flow | **3a1.1 (Ngoại lệ của luồng Hủy đơn):** Đơn hàng đã chuyển sang trạng thái \"Đang giao\" (Shipped) hoặc \"Đã giao\" (Delivered). 3a1.2. Hệ thống tự động vô hiệu hóa (disable) hoặc ẩn nút \"Hủy đơn hàng\", khách hàng không thể thực hiện thao tác này. |
| Business Rules | \- **BR06-1 (QĐ_KH6):** Trạng thái trên thanh tiến trình (Order Stepper) được cập nhật tự động và đồng bộ từ Đơn vị vận chuyển. Khách hàng không cần rời khỏi sàn để tra cứu. \- **BR06-2:** Chỉ cho phép Khách hàng hủy đơn khi đơn hàng đang ở trạng thái \"Pending\" hoặc \"Placed\". |
| Non-Functional Requirement | \- **NFR06-1:** Lịch sử đơn hàng của khách hàng phải được lưu trữ trên hệ thống tối thiểu 12 tháng để tra cứu đối soát. |

### Use case 7

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC07 |
| Use Case Name | Tương tác Chatbot AI |
| Description | Là một Người dùng, tôi muốn chat bằng ngôn ngữ tự nhiên với trợ lý ảo AI để tìm hiểu tổng quan về hệ thống, xu hướng sản phẩm, hoặc tra cứu tình trạng đơn hàng/giỏ hàng cá nhân. |
| Actor(s) | Khách vãng lai (Guest), Khách hàng (Customer) |
| Priority | Should Have |
| Trigger | Người dùng nhấp vào biểu tượng Chatbot nổi ở góc màn hình. |
| Pre-Condition(s) | Hệ thống AI Server đang hoạt động bình thường. |
| Post-Condition(s) | Người dùng nhận được câu trả lời chính xác dựa trên kho dữ liệu và ngữ cảnh định danh của hệ thống. |
| Basic Flow | 1\. Người dùng mở khung Chatbot AI. 2\. Người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên (Ví dụ: *\"Nền tảng này bán gì?\", \"Sản phẩm nào đang hot nhất?\",* hoặc *\"Đơn hàng của tôi đâu?\"*). 3\. Khách hàng nhấn Gửi. 4\. Hệ thống tiếp nhận, AI phân tích ý định (intent) của câu hỏi. 5\. Hệ thống nhận diện trạng thái đăng nhập và truy xuất dữ liệu phù hợp để trả lời. 6\. Chatbot phản hồi lại tin nhắn cho người dùng kèm theo thông tin chi tiết. |
| Alternative Flow | 2a. Người dùng đang ở trong trang Chi tiết Sản phẩm và mở Chatbot hỏi về sản phẩm đó (Ví dụ: *\"Sản phẩm này tôi được giảm giá bao nhiêu?\"*). 2a1. Hệ thống AI tự động bắt ngữ cảnh của ID sản phẩm đang xem và trả về đúng thông số % giảm giá, màu sắc của sản phẩm đó. |
| Exception Flow | 5a. Phân quyền dữ liệu cá nhân: 5a1. Khách vãng lai (Guest) đặt câu hỏi liên quan đến dữ liệu cá nhân (Giỏ hàng, Đơn hàng, Ví xu). 5a2. Chatbot từ chối trả lời, yêu cầu Người dùng phải Đăng nhập và hiển thị kèm nút \"Đi đến trang Đăng nhập\". 5b. AI không hiểu câu hỏi: 5b1. Câu hỏi nằm ngoài phạm vi dữ liệu hệ thống (Ví dụ: Hỏi về thời tiết). 5b2. Hệ thống phản hồi lại thông báo từ chối khéo léo và hướng dẫn khách hàng hỏi lại các vấn đề liên quan đến mua sắm. |
| Business Rules | \- **BR07-1:** Nếu là Khách vãng lai, Chatbot chỉ được phép truy xuất kho dữ liệu Public (Tổng quan hệ thống, FAQ, Sản phẩm tìm kiếm nhiều nhất, Khuyến mãi). \- **BR07-2:** Nếu là Khách hàng (đã đăng nhập), Chatbot được cấp thêm quyền truy xuất dữ liệu Private (Đơn hàng, Giỏ hàng, Lịch sử) thuộc sở hữu của chính User đó. |
| Non-Functional Requirement | \- **NFR07-1:** Thời gian phản hồi của Chatbot AI (AI processing time) không được vượt quá 3 giây để đảm bảo tính thời gian thực. |

### Use case 8

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC08 |
| Use Case Name | Đánh giá sản phẩm |
| Description | Là một Khách hàng, tôi muốn chấm điểm và viết nhận xét (review) cho món hàng mình đã mua để phản hồi chất lượng cho Người bán và những người mua sau. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Should Have |
| Trigger | Khách hàng nhấn nút \"Viết đánh giá\" (Write review) tại màn hình Lịch sử đơn hàng. |
| Pre-Condition(s) | Khách hàng đã đăng nhập và Đơn hàng chứa sản phẩm đó phải ở trạng thái \"Đã giao\" (DELIVERED). |
| Post-Condition(s) | Bài đánh giá được lưu lại và hiển thị công khai trên trang chi tiết sản phẩm. |
| Basic Flow | 1\. Khách hàng chọn lệnh Viết đánh giá cho một sản phẩm đã nhận. 2\. Hệ thống hiển thị biểu mẫu (Form) đánh giá. 3\. Khách hàng chọn số sao mong muốn (Rating từ 1 đến 5 sao). 4\. Khách hàng nhập nội dung bình luận (Review text) và tải lên các hình ảnh thực tế của sản phẩm. 5\. Khách hàng chọn lệnh \"Gửi đánh giá\". 6\. Hệ thống lưu hình ảnh lên Cloudinary, ghi nhận đánh giá vào cơ sở dữ liệu. 7\. Hệ thống thông báo thành công và cập nhật lại điểm đánh giá trung bình của sản phẩm đó. |
| Alternative Flow | 1a. Khách hàng muốn xóa bài đánh giá: 1a1. Khách hàng truy cập lại bài đánh giá mình đã viết và chọn lệnh \"Xóa\". 1a2. Hệ thống yêu cầu xác nhận. Khách hàng đồng ý. 1a3. Hệ thống xóa bài đánh giá và tính lại điểm sao trung bình. |
| Exception Flow | **4a.** Khách hàng không chọn số sao (để trống Rating). 4a1. Hệ thống báo lỗi \"Vui lòng chọn số sao đánh giá\" và chặn thao tác gửi. **6a.** Quá trình tải ảnh lên máy chủ thất bại do sai định dạng hoặc quá dung lượng. 6a1. Hệ thống báo lỗi và yêu cầu tải lại hình ảnh. |
| Business Rules | \- **BR08-1 (QĐ_KH7):** Hệ thống chỉ kích hoạt nút \"Viết đánh giá\" khi trạng thái đơn hàng của sản phẩm đó chính xác là DELIVERED. Khách chưa mua hoặc chưa nhận hàng tuyệt đối không được đánh giá. |
| Non-Functional Requirement | \- **NFR08-1:** Hệ thống xử lý cập nhật lại điểm đánh giá trung bình (Average Rating) của sản phẩm ngay lập tức sau khi Submit. |

### Use case 9

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC09 |
| Use Case Name | Yêu cầu Trả hàng & Hoàn tiền |
| Description | Là một Khách hàng, tôi muốn yêu cầu hệ thống cho trả lại hàng lỗi và hoàn lại tiền, đồng thời có thể khiếu nại lên Ban Quản trị nếu bị Người bán làm khó dễ. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng nhấn nút \"Yêu cầu Trả hàng/Hoàn tiền\" tại đơn hàng. |
| Pre-Condition(s) | Đơn hàng phải ở trạng thái \"Đã giao\" (DELIVERED) và nằm trong thời gian quy định (Ví dụ: 7 ngày kể từ khi nhận). |
| Post-Condition(s) | Đơn hàng bị đóng băng dòng tiền đối soát và chuyển sang trạng thái chờ xử lý (RETURN_REQUESTED hoặc DISPUTED). |
| Basic Flow | 1\. Khách hàng truy cập lịch sử mua hàng, chọn một đơn hàng đủ điều kiện và chọn lệnh \"Yêu cầu trả hàng\". 2\. Hệ thống hiển thị biểu mẫu yêu cầu cung cấp minh chứng. 3\. Khách hàng điền lý do chi tiết (VD: Hàng lỗi, giao sai màu) và tải lên hình ảnh/video khui hàng. 4\. Khách hàng chọn gửi yêu cầu. 5\. Hệ thống xác nhận, chuyển trạng thái đơn hàng sang RETURN_REQUESTED và gửi thông báo cho Người bán xử lý. 6\. Tùy thuộc vào quyết định của Người bán, nếu chấp nhận, quy trình trả hàng diễn ra và Khách hàng được hoàn tiền về tài khoản. |
| Alternative Flow | 6a. Khiếu nại (Dispute): Nếu Người bán ấn từ chối yêu cầu trả hàng ở Bước 6. 6a1. Hệ thống báo kết quả từ chối về cho Khách hàng. 6a2. Khách hàng chọn lệnh \"Khiếu nại lên Admin\" (Escalate to Admin). 6a3. Hệ thống ghi nhận khiếu nại và đổi trạng thái đơn hàng sang DISPUTED (Đang tranh chấp). 6a4. Quản trị viên (Admin) sẽ vào làm trọng tài để phán quyết cuối cùng dựa trên các bằng chứng từ Khách và Seller. |
| Exception Flow | 1a. Đơn hàng đã quá thời hạn 7 ngày kể từ ngày nhận hàng. 1a1. Hệ thống ẩn nút \"Yêu cầu trả hàng\", khách hàng không thể thao tác. |
| Business Rules | \- **BR09-1 (QĐ_KH9):** Policy thời hạn hoàn trả được cấu hình là 7 ngày kể từ khi trạng thái là DELIVERED. Qua ngày thứ 8, hệ thống tự động khóa tính năng này. \- **BR09-2 (QĐ_KH10):** Khi đơn hàng bị đẩy lên trạng thái DISPUTED, quyết định của Admin trên hệ thống là kết quả bắt buộc cuối cùng. |
| Non-Functional Requirement | \- **NFR09-1:** Việc lưu trữ hình ảnh/video bằng chứng (Evidences) phải được tối ưu nén trên Cloudinary để giảm tải database máy chủ. |

### Use case 10

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC10 |
| Use Case Name | Cập nhật thông tin cá nhân |
| Description | Là một Khách hàng, tôi muốn cập nhật thông tin cá nhân của mình, để đảm bảo hệ thống lưu trữ đúng dữ liệu liên lạc phục vụ cho quá trình xác thực và chăm sóc khách hàng. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng truy cập vào Menu \"Tài khoản của tôi\" và chọn tab \"Hồ sơ cá nhân\" (Profile). |
| Pre-Condition(s) | Khách hàng đã đăng nhập vào hệ thống. |
| Post-Condition(s) | Thông tin cá nhân mới được lưu trữ thành công vào cơ sở dữ liệu. |
| Basic Flow | 1\. Khách hàng truy cập tab \"Hồ sơ cá nhân\". 2\. Hệ thống truy xuất dữ liệu hiện tại và hiển thị trên biểu mẫu (Form). 3\. Khách hàng thực hiện chỉnh sửa các trường thông tin mong muốn (Họ tên, Số điện thoại). 4\. Khách hàng nhấn nút \"Lưu thay đổi\" 5\. Hệ thống kiểm tra tính hợp lệ của dữ liệu đầu vào. 6\. Hệ thống ghi nhận thông tin mới vào cơ sở dữ liệu. 7\. Hệ thống hiển thị thông báo \"Cập nhật thành công\" và làm mới lại dữ liệu hiển thị. |
| Alternative Flow | *Không có nhánh rẽ phức tạp, người dùng thao tác trực tiếp trên biểu mẫu.* |
| Exception Flow | 5a. Khách hàng bỏ trống trường thông tin bắt buộc (Họ tên) hoặc nhập sai định dạng Số điện thoại. 5a1. Hệ thống hiển thị thông báo lỗi bôi đỏ tại trường tương ứng và chặn lệnh lưu. *Use Case quay lại bước 3.* |
| Business Rules | \- **BR10-1:** Khách hàng không được phép thay đổi Email đăng nhập. Trường Email được đặt ở chế độ Read-only (Chỉ đọc) trên giao diện. |
| Non-Functional Requirement | \- **NFR10-1:** Thông tin thay đổi phải được cập nhật ngay lập tức lên Header (Khu vực hiển thị Tên Avatar) nhờ cơ chế quản lý trạng thái (Redux Toolkit) mà không cần tải lại trang. |

### Use case 11

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC11 |
| Use Case Name | Quản lý Địa chỉ |
| Description | Là một Khách hàng, tôi muốn quản lý danh sách đa địa chỉ nhận hàng, để có thể chọn nhanh địa chỉ phù hợp tại bước Thanh toán mà không phải nhập tay lại từ đầu. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Must Have |
| Trigger | Khách hàng truy cập vào tab \"Sổ địa chỉ\" (Saved Addresses) trong phần Quản lý tài khoản. |
| Pre-Condition(s) | Khách hàng đã đăng nhập. |
| Post-Condition(s) | Địa chỉ được Thêm mới, Cập nhật hoặc Xóa thành công khỏi Sổ địa chỉ của khách hàng. |
| Basic Flow | 1\. Khách hàng truy cập tab \"Sổ địa chỉ\". 2\. Hệ thống hiển thị danh sách các Thẻ địa chỉ (Address Cards) hiện có. 3. Khách hàng chọn một thao tác: - **Thêm mới:** Nhấn \"Thêm địa chỉ mới\", hệ thống hiển thị form trống. - **Cập nhật:** Nhấn \"Chỉnh sửa\" tại một thẻ địa chỉ, hệ thống hiển thị form chứa sẵn dữ liệu cũ. - **Xóa:** Nhấn icon \"Xóa\" tại một thẻ địa chỉ. 4\. Khách hàng điền/sửa thông tin (Tên, SĐT, Tỉnh/Thành, Quận/Huyện, Chi tiết) và xác nhận lệnh. 5\. Hệ thống kiểm tra dữ liệu, ghi nhận thay đổi vào cơ sở dữ liệu. 6\. Hệ thống hiển thị thông báo thành công và tự động cập nhật lại danh sách Sổ địa chỉ. |
| Alternative Flow | **1a.** Hành động \"Thêm địa chỉ mới\" cũng có thể được kích hoạt trực tiếp từ màn hình Đặt hàng và Thanh toán (Checkout) thay vì phải vào Quản lý tài khoản. |
| Exception Flow | **5a.** Khách hàng thực hiện \"Thêm/Sửa\" nhưng bỏ trống các trường bắt buộc. **5a1.** Hệ thống báo lỗi bôi đỏ tại các trường chưa nhập và chặn lệnh lưu. **3a.** Khách hàng Xóa địa chỉ duy nhất đang tồn tại hoặc địa chỉ đang được sử dụng cho một đơn hàng chưa giao xong. **3a1.** Hệ thống hiển thị cảnh báo từ chối xóa để đảm bảo an toàn tiến trình giao hàng. |
| Business Rules | \- **BR11-1 (QĐ_KH14):** Sổ địa chỉ cho phép lưu nhiều bản ghi. Danh sách này sẽ được gọi ra dưới dạng các Address Card tại bước Thanh toán (UC05). |
| Non-Functional Requirement | \- **NFR11-1:** Các Form nhập liệu Tỉnh/Thành/Phường/Xã nên được thiết kế dưới dạng Dropdown Select gọi từ API hành chính để đồng bộ chuẩn dữ liệu. |

### Use case 12

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC12 |
| Use Case Name | Theo dõi ví xu |
| Description | Là một Khách hàng, tôi muốn theo dõi biến động số dư Ví Xu của mình, để biết chính xác số lượng xu tích lũy có thể dùng để giảm giá cho các đơn đặt hàng tiếp theo. |
| Actor(s) | Khách hàng (Customer) |
| Priority | Should Have |
| Trigger | Khách hàng truy cập vào tab \"Ví Xu\" (Coins) trong phần Quản lý tài khoản. |
| Pre-Condition(s) | Khách hàng đã đăng nhập. |
| Post-Condition(s) | Khách hàng nắm bắt được chính xác số dư hiện tại và lịch sử nhận/trừ xu. |
| Basic Flow | 1\. Khách hàng truy cập tab \"Ví Xu\". 2\. Hệ thống gửi yêu cầu truy xuất dữ liệu từ máy chủ. 3\. Hệ thống hiển thị Thẻ tổng quan chứa: Tổng số dư Xu hiện tại (Available Coins). 4\. Hệ thống hiển thị Danh sách lịch sử biến động (Transaction History) bên dưới, bao gồm các thông tin: - Trạng thái: Nhận (Earned) hoặc Trừ (Spent). - Số lượng Xu biến động (+/-). - Thời gian thực hiện và Mã đơn hàng tham chiếu. 5\. Khách hàng tra cứu thông tin (Thao tác chỉ đọc). |
| Alternative Flow | *Không có.* |
| Exception Flow | *Xử lý lỗi rớt mạng hoặc quá thời gian tải dữ liệu (Timeout).* |
| Business Rules | \- **BR12-1 (QĐ_KH11):** Giao diện Ví Xu ở chế độ Read-only (Chỉ Đọc) đối với Khách hàng. Việc cộng/trừ xu được hệ thống tự động kích hoạt dựa trên trạng thái Đơn hàng. |
| Non-Functional Requirement | \- **NFR12-1:** Lịch sử biến động ví xu cần được sắp xếp theo thứ tự thời gian mới nhất lên đầu (Sort DESC by Time). |

### Use case 13

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC13 |
| Use Case Name | Chat trực tuyến (Real-time) |
| Description | Là một Khách hàng hoặc Người bán, tôi muốn trò chuyện trực tiếp với đối phương theo thời gian thực để trao đổi chi tiết, tư vấn và giải đáp thắc mắc về sản phẩm/đơn hàng. |
| Actor(s) | Khách hàng (Customer), Người bán (Seller) |
| Priority | Must Have |
| Trigger | Khách hàng nhấn vào nút \"Chat với Người bán\" tại trang sản phẩm, hoặc Người bán nhấn nút \"Phản hồi Khách hàng\" từ Bảng điều khiển (Dashboard). |
| Pre-Condition(s) | Cả hai tác nhân đều đã đăng nhập thành công vào hệ thống. |
| Post-Condition(s) | Nội dung đoạn chat được hệ thống lưu trữ làm lịch sử và hiển thị đồng bộ trên thiết bị của cả hai bên. |
| Basic Flow | 1\. Khách hàng nhấn vào biểu tượng \"Chat\" trên trang chi tiết sản phẩm hoặc đơn hàng. 2\. Hệ thống thiết lập kênh kết nối WebSockets và hiển thị cửa sổ hộp thoại trò chuyện. 3\. Khách hàng nhập nội dung tin nhắn và nhấn \"Gửi\". 4\. Hệ thống lưu tin nhắn vào cơ sở dữ liệu và ngay lập tức đẩy (push) tin nhắn đó đến Bảng điều khiển của Người bán tương ứng. 5\. Người bán nhận được thông báo tin nhắn mới, mở hộp thoại trò chuyện và gõ câu trả lời. 6\. Hệ thống đẩy tin nhắn phản hồi, hiển thị tức thì (Real-time) lên màn hình của Khách hàng mà không cần phải tải lại trang web. |
| Alternative Flow | 1a. Người bán chủ động khởi tạo cuộc trò chuyện với Khách hàng từ màn hình Quản lý đơn hàng (ví dụ: để thông báo hết màu/size). Các bước gửi và nhận tin nhắn tiếp theo diễn ra tương tự như luồng Basic Flow. |
| Exception Flow | 2a. Lỗi mất kết nối mạng: 2a1. Hệ thống không thể duy trì phiên WebSockets. Hệ thống hiển thị thông báo \"Đang mất kết nối\...\" và tạm thời làm mờ nút \"Gửi\" để ngăn chặn mất dữ liệu. 2a2. Khi có mạng trở lại, hệ thống tự động kết nối (Re-connect) và fetch lại các tin nhắn bị nhỡ (nếu có). |
| Business Rules | \- **BR13-1 (QĐ_KH13 / QĐ_SL6):** Cuộc trò chuyện mang tính chất riêng tư giữa một Khách hàng cụ thể và Gian hàng cụ thể của sản phẩm đó. |
| Non-Functional Requirement | \- **NFR13-1 (Cốt lõi):** Tính năng bắt buộc sử dụng giao thức **WebSockets** (Cụ thể là Spring WebSockets + STOMP ở Back-end và thư viện SockJS/stompjs ở Front-end) để duy trì kết nối liên tục thay vì dùng HTTP Requests thông thường. Độ trễ tin nhắn (Latency) phải đảm bảo dưới 1 giây. |

### Use case 14

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC14 |
| Use Case Name | Quản lý Hồ sơ và Gian hàng |
| Description | Là một Người bán, tôi muốn thiết lập thông tin doanh nghiệp, tài khoản ngân hàng và địa chỉ kho để hệ thống có thể đối soát dòng tiền và lấy hàng giao cho khách. |
| Actor(s) | Người bán (Seller) |
| Priority | Must Have |
| Trigger | Người bán truy cập vào tab \"Profile\" (Hồ sơ). |
| Pre-Condition(s) | Người bán đã đăng nhập thành công vào hệ thống. |
| Post-Condition(s) | Hồ sơ được cập nhật thành công, dữ liệu được ghi nhận vào cơ sở dữ liệu. |
| Basic Flow | 1\. Người bán truy cập trang cấu hình Hồ sơ gian hàng. 2\. Hệ thống hiển thị Dashboard với các khối thông tin: Thông tin Doanh nghiệp (Business Details), Chi tiết Ngân hàng (Bank Details), và Địa chỉ Kho hàng (Pickup Address). 3\. Người bán nhập hoặc chỉnh sửa các trường thông tin cần thiết (Tên Shop, Mã số thuế GST, Tên chủ tài khoản, Số tài khoản ngân hàng, Địa chỉ kho lấy hàng). 4\. (Tùy chọn) Người bán tải lên hình ảnh Logo và Banner cho gian hàng. 5\. Người bán nhấn \"Lưu thay đổi\". 6\. Hệ thống kiểm tra dữ liệu, upload ảnh lên Cloudinary và lưu thông tin vào cơ sở dữ liệu. 7\. Hệ thống thông báo \"Cập nhật thành công\" và làm mới dữ liệu trên màn hình. |
| Alternative Flow | 7a. Xem trước gian hàng (Preview Storefront): Sau khi lưu thành công ở bước 7, Người bán nhấn vào nút \"Xem gian hàng\". Hệ thống mở một thẻ trình duyệt mới hiển thị giao diện mặt tiền công khai của Shop (bao gồm Logo, Banner vừa đổi và danh sách các sản phẩm đang bán).). |
| Exception Flow | 5a. Người bán bỏ trống các thông tin quan trọng để đối soát như Số tài khoản ngân hàng hoặc Mã số thuế (GST). 5a1. Hệ thống báo lỗi validation ngay dưới trường nhập liệu và chặn lệnh lưu. *Use Case quay lại bước 3.* |
| Business Rules | \- **BR14-1 (QĐ_SL1):** Seller bắt buộc phải cung cấp Mã số thuế doanh nghiệp (GST), địa chỉ kho lấy hàng (Pickup address) và thông tin ngân hàng hợp lệ để hệ thống có cơ sở trả tiền đối soát (Payout) sau khi hoàn tất đơn hàng. |
| Non-Functional Requirement | \- **NFR14-1:** Các tệp tin Logo và Banner của Shop phải được đẩy trực tiếp lên Cloudinary API từ phía Client (sử dụng FormData) để giảm tải băng thông cho Server backend. |

### Use case 15

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC15 |
| Use Case Name | Quản lý Kho sản phẩm |
| Description | Là một Người bán, tôi muốn đăng tải sản phẩm mới hoặc cập nhật giá bán, số lượng tồn kho để hàng hóa có thể được hiển thị tới Khách hàng. |
| Actor(s) | Người bán (Seller) |
| Priority | Must Have |
| Trigger | Người bán chọn \"Add Product\" (Thêm sản phẩm) hoặc \"Update Stock\" (Cập nhật kho) tại trang Quản lý sản phẩm. |
| Pre-Condition(s) | Tài khoản Người bán phải ở trạng thái được phê duyệt (ACTIVE). |
| Post-Condition(s) | Sản phẩm được lưu vào hệ thống và chuyển trạng thái thành chờ kiểm duyệt hoặc được hiển thị trực tiếp (tuỳ chính sách Admin). |
| Basic Flow | 1\. Người bán chọn lệnh \"Thêm sản phẩm mới\". 2\. Hệ thống hiển thị biểu mẫu thông tin Sản phẩm. 3\. Người bán chọn Danh mục phân cấp cho sản phẩm (Level 1, Level 2, Level 3). 4\. Người bán điền Tiền gốc (MRP) và Giá bán thực tế (Selling Price). Hệ thống tự động tính toán % Giảm giá dựa trên 2 mức giá này. 5\. Người bán cung cấp Tên sản phẩm, Mô tả, Kích cỡ, Màu sắc và Số lượng tồn kho (Quantity). 6\. Người bán tải lên các hình ảnh thực tế của sản phẩm (tối đa 4-5 ảnh). 7\. Người bán nhấn lệnh \"Đăng sản phẩm\". 8\. Hệ thống ghi nhận sản phẩm vào cơ sở dữ liệu. Sản phẩm mặc định được đẩy vào trạng thái \"Chờ duyệt\" (Pending Approval). |
| Alternative Flow | **1a. Cập nhật tồn kho (Update Stock):** 1a1. Tại lưới danh sách Sản phẩm hiện có, Người bán chỉnh sửa trực tiếp con số tồn kho và nhấn icon \"Update\". 1a2. Hệ thống cập nhật nhanh tồn kho hiện tại (In-stock) và hiển thị lại lưới mà không cần tải trang. |
| Exception Flow | **4a.** Người bán điền Giá bán thực tế (Selling Price) LỚN HƠN Giá gốc (MRP). **4a1.** Hệ thống báo lỗi logic \"Giá bán không được lớn hơn giá gốc\" và chặn nút lưu. **6a.** Quá trình upload ảnh lên máy chủ Cloudinary thất bại do lỗi mạng. **6a1.** Vòng xoay tiến trình (Circular Progress) báo lỗi và yêu cầu Người bán chọn lại ảnh. |
| Business Rules | \- **BR15-1 (QĐ_SL2):** Mỗi sản phẩm bắt buộc phải được gắn vào đúng 1 Cây danh mục Level 3 (Ví dụ: Men -\> Topwear -\> T-Shirt). Khách hàng sẽ dùng ID danh mục này để lọc sản phẩm tại Trang chủ (UC01). \- **BR15-2 (QĐ_AD11):** Để bảo vệ nền tảng, mọi sản phẩm tạo mới đều ở trạng thái ẩn (PENDING). Chỉ khi Admin (Quản trị viên) phê duyệt ở UC20, sản phẩm mới được hiển thị công khai trên gian hàng. |
| Non-Functional Requirement | \- **NFR15-1:** Khi người bán thay đổi giá tiền ở bước 4, % Giảm giá phải được tự động tính toán bằng JavaScript trên giao diện (Client-side) ngay lập tức theo thời gian thực. |

### Use case 16

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC16 |
| Use Case Name | Xử lý Đơn hàng & Vận chuyển |
| Description | Là một Người bán, tôi muốn tiếp nhận đơn đặt hàng, xác nhận đóng gói và kết xuất mã vận đơn để tiến hành giao hàng cho khách. |
| Actor(s) | Người bán (Seller) |
| Priority | Must Have |
| Trigger | Người bán truy cập vào tab \"Orders\" (Đơn hàng) để xem các đơn hàng vừa được khách đặt. |
| Pre-Condition(s) | Có đơn hàng phát sinh thuộc về mã gian hàng của Seller đó. |
| Post-Condition(s) | Trạng thái đơn hàng thay đổi, Mã vận đơn được sinh ra để hai bên cùng tra cứu. |
| Basic Flow | 1\. Người bán truy cập tab Quản lý đơn hàng. Hệ thống hiển thị danh sách các đơn hàng theo lưới (Có thể phân bộ lọc: Mới đặt, Chờ lấy hàng, Đang giao\...). 2\. Người bán nhấn vào một đơn hàng Mới (Trạng thái: PLACED) và chọn hành động \"Xác nhận đơn\" (Confirm). 3\. Trạng thái đơn hàng chuyển sang CONFIRMED (Đã xác nhận). 4\. Người bán tiến hành đóng gói, sau đó chọn lệnh \"Đẩy đơn vận chuyển\" (Ship Order). 5\. Hệ thống Back-end tự động gọi API giao tiếp với đối tác vận chuyển (VD: GHTK, Grab) truyền đi thông tin Khách hàng và Địa chỉ kho Seller (Pickup Address). 6\. API vận chuyển trả về Mã vận đơn (Tracking ID). Hệ thống lưu mã này vào Đơn hàng, đổi trạng thái thành SHIPPED (Đang giao). 7\. Người bán chọn lệnh \"In phiếu giao hàng\", hệ thống tự động kết xuất (Generate) tài liệu PDF mã vạch để Seller dán lên gói hàng. |
| Alternative Flow | 6a. Cập nhật qua Webhook: Sau khi hàng được đẩy đi ở bước 6, người bán không cần tác động thủ công nữa. Khi Shipper giao hàng thành công, hệ thống của ĐVVC sẽ tự bắn API (Webhook) về máy chủ E-commerce để đổi trạng thái đơn hàng sang DELIVERED (Đã giao). |
| Exception Flow | 5a. API kết nối với Đơn vị vận chuyển gặp sự cố (Timeout) hoặc Địa chỉ kho lấy hàng của Seller không hợp lệ. 5a1. Hệ thống báo lỗi \"Kết nối hãng vận chuyển thất bại. Vui lòng thử lại sau\" và giữ nguyên trạng thái đơn ở mức CONFIRMED. |
| Business Rules | \- **BR16-1 (QĐ_SL3):** Việc kết nối tạo mã vận đơn (Tracking ID) là quy trình xử lý tự động (Automation). Seller không được tự nhập tay mã tracking để tránh gian lận đối soát phí ship. \- **BR16-2 (QĐ_SL4):** Phiếu giao hàng (Vận đơn PDF) bắt buộc chứa Barcode/QR Code của ĐVVC để tài xế có thể dùng máy quét tít mã lấy hàng. |
| Non-Functional Requirement | \- **NFR16-1:** Vì lệnh gọi API ra bên thứ 3 (Hãng vận chuyển) có độ trễ, giao diện phải hiển thị Loading Spinner chặn thao tác tay của Seller tránh việc ấn \"Đẩy đơn\" liên tục sinh ra nhiều mã vận đơn rác. |

### Use case 17

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC17 |
| Use Case Name | Xử lý Yêu cầu Hoàn trả |
| Description | Là một Người bán, tôi muốn xem xét lý do và các minh chứng để đưa ra quyết định chấp nhận hoặc từ chối yêu cầu trả hàng/hoàn tiền từ Khách hàng. |
| Actor(s) | Người bán (Seller) |
| Priority | Must Have |
| Trigger | Người bán nhận được thông báo hoặc truy cập trực tiếp vào tab \"Yêu cầu hoàn trả\" (Return Requests). |
| Pre-Condition(s) | Có ít nhất một đơn hàng đang ở trạng thái yêu cầu hoàn trả (RETURN_REQUESTED). |
| Post-Condition(s) | Yêu cầu được giải quyết. Trạng thái đơn hàng được cập nhật và hệ thống tự động xử lý các bước hoàn tiền/đối soát tiếp theo. |
| Basic Flow | 1\. Người bán truy cập danh sách các yêu cầu trả hàng từ Khách hàng. 2\. Hệ thống truy xuất và hiển thị danh sách các đơn hàng đang ở trạng thái RETURN_REQUESTED. 3\. Người bán chọn xem chi tiết một yêu cầu, đọc lý do và xem các hình ảnh/video minh chứng do Khách hàng cung cấp. 4\. Người bán quyết định chọn lệnh **\"**Chấp nhận\" (Accept). 5\. Hệ thống gửi thông báo cho Khách hàng yêu cầu gửi trả hàng về kho. 6\. Sau khi Người bán nhận lại hàng và bấm xác nhận, hệ thống tự động gọi API hoàn tiền trả về tài khoản Khách hàng và cập nhật đơn hàng thành trạng thái REFUNDED. |
| Alternative Flow | 4a. Người bán chọn lệnh \"Từ chối\" (Reject): 4a1. Người bán bấm \"Từ chối\" đối với yêu cầu của Khách hàng. 4a2. Hệ thống hiển thị khung nhập liệu yêu cầu nhập lý do từ chối. 4a3. Người bán điền lý do và xác nhận. 4a4. Hệ thống ghi nhận quyết định, thông báo cho Khách hàng lý do từ chối và mở ra khả năng cho phép Khách hàng Khiếu nại (Dispute) lên Admin. |
| Exception Flow | 1a. Vượt quá thời hạn xử lý: 1a1. Quá thời hạn 3 ngày kể từ khi Khách hàng tạo yêu cầu mà Người bán chưa có phản hồi, hệ thống hiển thị cảnh báo vi phạm thời gian xử lý (SLA) đối với gian hàng đó. |
| Business Rules | \- **BR17-1 (QĐ_SL5):** Seller có tối đa 3 ngày để phản hồi yêu cầu. Trong trường hợp Từ chối, Seller bắt buộc phải ghi rõ lý do từ chối để Khách hàng (và Admin sau này) có cơ sở đối chứng. |
| Non-Functional Requirement | \- **NFR17-1:** Giao diện xem hình ảnh/video minh chứng tải từ Cloudinary phải hỗ trợ Zoom (phóng to) và phát media mượt mà trực tiếp trên Dashboard mà không cần tải file về máy. |

### Use case 18

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC18 |
| Use Case Name | Theo dõi Đối soát & Doanh thu |
| Description | Là một Người bán, tôi muốn xem biểu đồ doanh thu tổng quan, theo dõi lịch sử dòng tiền và xuất dữ liệu ra file Excel để thực hiện nghiệp vụ kế toán nội bộ. |
| Actor(s) | Người bán (Seller) |
| Priority | Should Have |
| Trigger | Người bán truy cập vào trang \"Bảng điều khiển\" (Dashboard) hoặc tab \"Giao dịch\" (Transactions). |
| Pre-Condition(s) | Người bán đã đăng nhập thành công vào hệ thống. |
| Post-Condition(s) | Dữ liệu thống kê được tính toán chính xác và kết xuất thành file báo cáo thành công. |
| Basic Flow | 1\. Người bán truy cập trang Bảng điều khiển tổng quan. 2\. Hệ thống tự động tổng hợp và tính toán các chỉ số kinh doanh theo thời gian thực. 3\. Hệ thống hiển thị các thẻ thống kê tổng thể bao gồm: Tổng thu nhập (Total Earning), Tổng số đơn (Total Orders), Đơn bị hủy (Canceled Orders) và Tổng hoàn tiền (Total Refund). 4\. Hệ thống hiển thị Biểu đồ doanh thu trực quan (Earning graphs) phân bổ theo ngày, tuần, hoặc tháng. 5\. Người bán truy cập tab \"Lịch sử giao dịch\" (Transactions) để xem đối soát chi tiết dòng tiền của từng đơn hàng cụ thể. |
| Alternative Flow | 5a. Xuất báo cáo dữ liệu Excel: 5a1. Tại màn hình Giao dịch/Báo cáo, Người bán chọn lệnh \"Xuất báo cáo\" (Export). 5a2. Hệ thống truy xuất dữ liệu từ Database, định dạng lại thành cấu trúc tệp tin .xlsx 5a3. Hệ thống tự động tải file báo cáo Excel xuống thiết bị của Người bán. |
| Exception Flow | 2a. Hệ thống chưa phát sinh giao dịch: 2a1. Nếu Seller là gian hàng mới chưa có đơn hàng nào, hệ thống không báo lỗi mà chỉ hiển thị dữ liệu \"0\" trên các thẻ và hiển thị trạng thái \"Empty state\" (Chưa có dữ liệu) tại biểu đồ đồ thị. |
| Business Rules | \- **BR18-1 (CT_SL1):** Báo cáo phải tự động tổng hợp chính xác các chỉ số từ tất cả các đơn hàng thuộc quyền sở hữu của Seller. \- **BR18-2 (CT_SL2):** Khi một đơn hàng hoàn tiền thành công ở UC17, khoản tiền này phải được tự động trừ khỏi Tổng thu nhập (Total Earning) và được cộng dồn vào thống kê Tổng hoàn tiền (Total Refund) để việc đối soát dòng tiền luôn minh bạch. |
| Non-Functional Requirement | \- **NFR18-1 (QĐ_SL7):** File Excel được kết xuất phải đảm bảo đúng định dạng bảng tính (Spreadsheet), không bị lỗi font chữ Unicode (tiếng Việt) để Người bán dễ dàng làm việc với các phần mềm kế toán. |

### Use case 19

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC19 |
| Use Case Name | Kiểm duyệt Người bán |
| Description | Là Quản trị viên, tôi muốn xét duyệt hồ sơ đăng ký gian hàng hoặc xử lý vi phạm để duy trì môi trường kinh doanh minh bạch và chất lượng trên sàn. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Must Have |
| Trigger | Admin truy cập vào tab \"Sellers\" trên Admin Dashboard. |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Trạng thái của tài khoản Seller được thay đổi (ACTIVE, SUSPENDED, hoặc BANNED). |
| Basic Flow | 1\. Admin truy cập danh sách quản lý Người bán. 2\. Hệ thống hiển thị lưới danh sách, Admin lọc theo trạng thái PENDING_VERIFICATION (Chờ xác minh). 3\. Admin nhấp vào xem chi tiết hồ sơ của một Người bán (bao gồm thông tin GST, địa chỉ kho, số tài khoản ngân hàng). 4\. Admin kiểm tra tính hợp lệ của thông tin và chọn lệnh \"Approve\" (Phê duyệt). 5\. Hệ thống thay đổi trạng thái tài khoản thành ACTIVE. 6\. Hệ thống tự động gửi Email thông báo chúc mừng đến Người bán và cho phép họ bắt đầu đăng sản phẩm. |
| Alternative Flow | 4a. Quản lý vi phạm (Đình chỉ / Cấm vĩnh viễn): 4a1. Tại danh sách các Seller đang hoạt động (ACTIVE), Admin phát hiện gian hàng vi phạm chính sách. 4a2. Admin chọn lệnh đổi trạng thái sang Suspend (Đình chỉ tạm thời) hoặc Ban (Cấm vĩnh viễn). 4a3. Hệ thống khóa quyền đăng nhập của Seller đó và ẩn toàn bộ sản phẩm của gian hàng khỏi Trang chủ. |
| Exception Flow | 4b. Hồ sơ Seller đăng ký thiếu các thông tin cốt lõi (Không có GST hoặc sai định dạng tài khoản ngân hàng). 4b1. Admin chọn lệnh \"Reject\" (Từ chối), ghi rõ lý do. Hệ thống gửi email yêu cầu Seller bổ sung thông tin. |
| Business Rules | \- **BR19-1 (QĐ_AD1):** Chỉ những Seller có trạng thái ACTIVE mới được quyền truy cập vào Seller Dashboard và đăng tải sản phẩm. |
| Non-Functional Requirement | \- **NFR19-1:** Mọi thao tác thay đổi trạng thái tài khoản Seller đều phải được lưu vết lại vào bảng Audit Log (Lịch sử hệ thống) để phục vụ kiểm toán sau này. |

### Use case 20

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC20 |
| Use Case Name | Kiểm duyệt Sản phẩm |
| Description | Là Quản trị viên, tôi muốn xem xét thông tin và hình ảnh của các sản phẩm mới do Seller đăng tải để phê duyệt hiển thị công khai hoặc từ chối. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Must Have |
| Trigger | Admin truy cập vào tab \"Kiểm duyệt Sản phẩm\" (Pending Products). |
| Pre-Condition(s) | Có ít nhất một sản phẩm mới được Seller đăng tải đang ở trạng thái PENDING. |
| Post-Condition(s) | Sản phẩm được phê duyệt hiển thị lên sàn hoặc bị trả về cho Seller. |
| Basic Flow | 1\. Admin mở danh sách các sản phẩm đang chờ duyệt. 2\. Admin chọn một sản phẩm và kiểm tra chi tiết: Tên, Mô tả, Cây danh mục, Giá bán, và Hình ảnh thực tế. 3\. Admin xác nhận sản phẩm không vi phạm quy định (không phải hàng giả, hàng cấm) và nhấn nút \"Approve\" (Phê duyệt). 4\. Hệ thống chuyển trạng thái sản phẩm sang PUBLISHED (Đã xuất bản). 5\. Sản phẩm ngay lập tức xuất hiện trên giao diện tìm kiếm và mặt tiền (Storefront) của Khách hàng. |
| Alternative Flow | 3a. Admin Từ chối sản phẩm: 3a1. Admin phát hiện sản phẩm sai danh mục hoặc ảnh kém chất lượng, chọn lệnh \"Reject\" (Từ chối). 3a2. Hệ thống yêu cầu nhập lý do từ chối. Admin điền lý do và xác nhận. 3a3. Sản phẩm bị đẩy về trạng thái REJECTED, Seller nhận được thông báo để chỉnh sửa lại. |
| Exception Flow | *Không có nhánh ngoại lệ phức tạp.* |
| Business Rules | \- **BR20-1 (QĐ_AD11):** Bắt buộc áp dụng cơ chế kiểm duyệt. Hệ thống tự động thiết lập trạng thái mặc định của mọi sản phẩm tạo mới là PENDING. Seller không thể tự ý Publish sản phẩm. |
| Non-Functional Requirement | \- **NFR20-1:** Giao diện duyệt sản phẩm phải được thiết kế tối ưu (Load ảnh nhanh từ Cloudinary) để Admin có thể duyệt hàng loạt (Mass approval) một cách nhanh chóng. |

### Use case 21

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC21 |
| Use Case Name | Quản lý Giao diện Trang chủ |
| Description | Là Quản trị viên, tôi muốn cấu hình động các Banner, Deals và lưới danh mục (Grid Categories) trên Trang chủ mà không cần phải nhờ Lập trình viên sửa code. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Should Have |
| Trigger | Admin truy cập vào tab \"Home Page Config\" trên Dashboard. |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Giao diện trang chủ (Homepage) của Khách hàng được tự động cập nhật layout mới nhất. |
| Basic Flow | 1\. Admin truy cập module quản lý Trang chủ. 2\. Hệ thống hiển thị các Section có thể tùy biến (Electric Categories, Grid Categories, Shop by Category). 3\. Admin chọn chỉnh sửa một Section (Ví dụ: Lưới danh mục). 4\. Admin cung cấp đường dẫn hình ảnh mới (Image URL), chọn Danh mục cấp 3 tương ứng (Ví dụ: Women -\> Footwear -\> Heels). 5\. Admin nhấn \"Cập nhật\". 6\. Hệ thống lưu cấu hình JSON vào cơ sở dữ liệu và thông báo cập nhật thành công. 7\. Admin tải lại Trang chủ Client, hình ảnh và liên kết danh mục mới ngay lập tức được áp dụng. |
| Alternative Flow | *Không có. Thao tác được xử lý trực tiếp trên biểu mẫu cấu hình (Form-based).* |
| Exception Flow | **4a.** Admin nhập URL hình ảnh sai định dạng hoặc bị lỗi. 4a1. Hình ảnh xem trước (Preview) bị vỡ. Hệ thống báo lỗi \"URL hình ảnh không hợp lệ\". |
| Business Rules | \- **BR21-1 (QĐ_AD3):** Dữ liệu cấu hình trang chủ (Homepage Data) phải được thiết kế thành một API public độc lập, để Client App (React) có thể gọi và render giao diện tự động dựa trên cấu hình mà Admin vừa lưu. |
| Non-Functional Requirement | \- **NFR21-1:** Trải nghiệm thay đổi giao diện theo nguyên tắc WYSIWYG (What You See Is What You Get). Các thay đổi cấu hình phải phản hồi tức thì dưới 1 giây. |

### Use case 22

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC22 |
| Use Case Name | Quản lý Chiến dịch Khuyến mãi |
| Description | Là Quản trị viên, tôi muốn phát hành và quản lý các Mã giảm giá (Coupons) và Khuyến mãi danh mục (Deals) để kích thích nhu cầu mua sắm chung trên toàn sàn. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Must Have |
| Trigger | Admin truy cập menu \"Coupons\" hoặc \"Deals\". |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Coupon hoặc Deal mới được phát hành, có sẵn để Khách hàng áp dụng. |
| Basic Flow | Trường hợp Tạo Coupon: 1\. Admin chọn lệnh \"Tạo Coupon mới\" (Add New Coupon). 2\. Hệ thống hiển thị biểu mẫu khởi tạo. 3\. Admin điền Mã Coupon (VD: DIWALI), % giảm giá (VD: 10%), Ngày bắt đầu, Ngày kết thúc và Giá trị đơn hàng tối thiểu (Min Order Value). 4\. Admin nhấn \"Khởi tạo\". 5\. Hệ thống lưu Coupon vào hệ thống và kích hoạt trạng thái (Active). |
| Alternative Flow | 1a. Cập nhật Deal giảm giá: 1a1. Admin chuyển sang tab \"Deals\", chọn một Deal hiện có (VD: Thời trang Nữ giảm 80%). 1a2. Admin thay đổi phần trăm giảm giá hoặc đổi ảnh banner của Deal đó. 1a3. Hệ thống lưu lại và làm mới (refresh) khu vực Today\'s Deal trên Trang chủ của người dùng. |
| Exception Flow | 3a. Admin chọn Ngày kết thúc (Validity End Date) diễn ra TRƯỚC Ngày bắt đầu (Validity Start Date). 3a1. Hệ thống báo lỗi logic thời gian và chặn thao tác lưu. |
| Business Rules | \- **BR22-1 (QĐ_AD4):** Các chiến dịch do Admin tạo áp dụng chung cho toàn nền tảng. Khi Khách hàng áp dụng Coupon này ở UC03/UC05, hệ thống sẽ đối soát để đảm bảo Seller không bị lỗ doanh thu. |
| Non-Functional Requirement | \- **NFR22-1:** Quá trình kiểm tra Coupon tại giỏ hàng của khách hàng đối với các Coupon do Admin tạo phải được truy vấn với tốc độ dưới 100ms. |

### Use case 23

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC23 |
| Use Case Name | Giải quyết khiếu nại (Disputes) |
| Description | Là Quản trị viên, tôi muốn làm trọng tài xử lý các đơn hàng đang có tranh chấp giữa Khách hàng và Người bán, đưa ra phán quyết cuối cùng và gọi lệnh hoàn tiền. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Must Have |
| Trigger | Admin truy cập vào tab \"Giải quyết Khiếu nại\" (Disputes). |
| Pre-Condition(s) | Có đơn hàng đang bị khóa ở trạng thái tranh chấp (DISPUTED) (kết quả từ UC09). |
| Post-Condition(s) | Đơn hàng được giải quyết xong, tiền được hoàn cho khách hoặc đẩy cho Seller. |
| Basic Flow | 1\. Admin xem danh sách các đơn hàng DISPUTED. 2\. Admin chọn một đơn hàng, hệ thống hiển thị song song 2 luồng dữ liệu: Lý do từ chối của Seller và Bằng chứng (Video/Ảnh) của Khách hàng. 3\. Admin đóng vai trò trọng tài để đánh giá minh chứng. 4\. Nếu lỗi thuộc về Seller, Admin chọn lệnh \"Chấp nhận khiếu nại - Hoàn tiền\". 5\. Hệ thống Backend tự động gọi API Refund của cổng thanh toán điện tử (VnPay/SePay/Momo) để trả lại tiền về thẻ của khách hàng. 6\. Trạng thái đơn hàng chuyển thành REFUNDED và khiếu nại đóng lại. |
| Alternative Flow | 4a. Phán quyết bảo vệ Người bán: 4a1. Admin nhận thấy bằng chứng của Khách hàng không hợp lý hoặc có dấu hiệu gian lận. 4a2. Admin chọn lệnh \"Từ chối khiếu nại\". 4a3. Hệ thống đóng khiếu nại, mở khóa dòng tiền đối soát và chuyển doanh thu của đơn hàng đó vào ví của Seller. |
| Exception Flow | 5a. Lỗi API Cổng thanh toán: 5a1. Hệ thống Backend gọi API Refund nhưng cổng Stripe/VnPay bị timeout hoặc từ chối lệnh. 5a2. Hệ thống báo lỗi cho Admin \"Hoàn tiền thất bại do lỗi cổng thanh toán\", giữ nguyên trạng thái DISPUTED để Admin thử lại sau. |
| Business Rules | \- **BR23-1 (QĐ_AD6):** Phán quyết của Admin là quyết định cao nhất và bắt buộc thực thi. Sau khi Admin đã xử lý, cả Khách hàng và Seller không được phép thao tác khiếu nại lại đối với đơn hàng này. \- **BR23-2 (QĐ_AD7):** Tuyệt đối không hoàn tiền thủ công. Việc trả tiền phải thông qua luồng tự động (Automated Refund API) để đảm bảo an toàn dòng tiền hệ thống. |
| Non-Functional Requirement | \- **NFR23-1:** Tính năng so sánh minh chứng (Evidences) phải cho phép tải và phát video trực tiếp trên dashboard của Admin mà không cần redirect qua trang web khác. |

### Use case 24

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC24 |
| Use Case Name | Quản lý Khách hàng |
| Description | Là Quản trị viên, tôi muốn xem danh sách khách hàng và quản lý trạng thái tài khoản để ngăn chặn các người dùng có hành vi vi phạm chính sách của nền tảng. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Should Have |
| Trigger | Admin truy cập vào tab \"Khách hàng\" (Customers) trên Admin Dashboard. |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Trạng thái tài khoản khách hàng được cập nhật thành công. |
| Basic Flow | 1\. Admin chọn tab \"Khách hàng\" trên trình đơn điều hướng. 2\. Hệ thống tải và hiển thị lưới danh sách toàn bộ khách hàng trên hệ thống. 3\. Admin tìm kiếm hoặc chọn một khách hàng cụ thể để xem chi tiết hồ sơ (Tên, Email, SĐT, Địa chỉ). 4\. (Tùy chọn) Nếu phát hiện tài khoản có hành vi gian lận (Ví dụ: lạm dụng mã giảm giá, boom hàng nhiều lần), Admin chọn lệnh \"Khóa tài khoản\" (Ban Account). 5\. Hệ thống xác nhận và thay đổi trạng thái tài khoản khách hàng thành BANNED. 6\. Hệ thống hiển thị thông báo \"Cập nhật trạng thái thành công\" và khóa quyền đăng nhập của người dùng này. |
| Alternative Flow | *Không có nhánh rẽ phức tạp, Admin thao tác trực tiếp trên lưới dữ liệu.* |
| Exception Flow | *Không có.* |
| Business Rules | \- **BR24-1 (QĐ_AD5):** Admin có quyền xem thông tin cơ bản của Khách hàng để hỗ trợ giải quyết sự cố, nhưng tuyệt đối không được xem mật khẩu của người dùng (mật khẩu đã được mã hóa BCrypt). |
| Non-Functional Requirement | \- **NFR24-1:** Bắt buộc áp dụng cơ chế Phân trang (Pagination) từ API Backend cho danh sách khách hàng để đảm bảo hiệu suất khi dữ liệu có hàng triệu user. |

### Use case 25

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC25 |
| Use Case Name | Cấu hình Thông số Tài chính |
| Description | Là Quản trị viên, tôi muốn thiết lập tỉ lệ quy đổi Ví Xu và mức phí nền tảng (Platform Fee) để điều tiết chính sách kinh doanh và lợi nhuận của sàn. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Must Have |
| Trigger | Admin truy cập vào tab \"Cấu hình Tài chính\" hoặc \"Cài đặt Hệ thống\" trên Dashboard. |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Các thông số tài chính mới được lưu lại và áp dụng ngay lập tức cho các giao dịch thanh toán và đối soát tiếp theo. |
| Basic Flow | 1\. Admin truy cập màn hình Cấu hình Thông số Tài chính. 2\. Hệ thống hiển thị biểu mẫu bao gồm 2 nhóm cấu hình chính: - Cấu hình Ví Xu (Reward Coins): Tỉ lệ kiếm xu, Tỉ giá tiêu xu, Hạn mức thanh toán bằng xu. - Cấu hình Phí sàn (Platform Fee): Phần trăm (%) phí trích xuất từ doanh thu mỗi đơn hàng thành công của Seller. 3\. Admin nhập các thông số mới (Ví dụ: Thu phí nền tảng 5%). 4\. Admin nhấn lệnh \"Lưu cấu hình\". 5\. Hệ thống ghi nhận các hằng số này vào cơ sở dữ liệu. 6\. Hệ thống hiển thị thông báo cập nhật thành công. |
| Alternative Flow | *Không có.* |
| Exception Flow | **3a.** Admin nhập số âm cho các trường Tỉ lệ phần trăm hoặc Tỉ giá quy đổi. 3a1. Hệ thống báo lỗi bôi đỏ \"Giá trị không hợp lệ, phải lớn hơn hoặc bằng 0\" và chặn lệnh lưu. *Use Case quay lại bước 3.* |
| Business Rules | \- **BR25-1 (QĐ_AD8):** Cấu hình xu phải đảm bảo 3 thông số: Tỉ lệ kiếm (VD: 1000đ = 1 Xu), Tỉ giá tiêu (1 Xu = 1đ), và Hạn mức tối đa (VD: Xu chỉ thanh toán tối đa 50% giá trị hóa đơn). \- **BR25-2 (QĐ_AD12):** % Phí nền tảng là biến cấu hình động, không fix cứng trong code. Thông số này sẽ được gọi ra để áp dụng vào Công thức CT_AD1 khi đối soát doanh thu thực nhận cho Seller ở UC18. |
| Non-Functional Requirement | \- **NFR25-1:** Các thay đổi về thông số tài chính chỉ áp dụng cho các Đơn hàng (Orders) được tạo ra SAU thời điểm cập nhật. Các đơn hàng trong quá khứ tuyệt đối không bị tính lại (để bảo toàn tính toàn vẹn của lịch sử kế toán). |

### Use case 26

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC26 |
| Use Case Name | Theo dõi Nhật ký (Audit Log) |
| Description | Là Quản trị viên, tôi muốn truy xuất và xem lại lịch sử các thao tác thay đổi dữ liệu quan trọng trên hệ thống để phục vụ công tác kiểm toán và rà soát lỗi. |
| Actor(s) | Quản trị viên (Admin) |
| Priority | Should Have |
| Trigger | Admin truy cập vào tab \"Nhật ký hệ thống\" (Audit Logs). |
| Pre-Condition(s) | Admin đã đăng nhập thành công. |
| Post-Condition(s) | Admin xem được toàn bộ lịch sử các thao tác quản trị một cách minh bạch. |
| Basic Flow | 1\. Admin truy cập vào trang Nhật ký hệ thống. 2\. Hệ thống truy xuất dữ liệu từ cơ sở dữ liệu và hiển thị danh sách các bản ghi nhật ký. 3\. Mỗi bản ghi log bao gồm các thông tin: Thời gian thực hiện (Timestamp), Tên tài khoản thực hiện (Actor), Hành động (Action - VD: Ban Seller, Phê duyệt sản phẩm, Đổi phí nền tảng). 4\. Admin sử dụng bộ lọc để tìm kiếm các hành động cụ thể theo khoảng thời gian hoặc theo phân hệ module. 5\. Admin tra cứu thông tin phục vụ kiểm toán. Thao tác xem hoàn tất. |
| Alternative Flow | *Không có.* |
| Exception Flow | *Không có nhánh ngoại lệ phức tạp, chủ yếu xử lý lỗi timeout nếu dữ liệu quá lớn.* |
| Business Rules | \- **BR26-1:** Bảng dữ liệu Nhật ký là dữ liệu Tuyệt đối Chỉ đọc (Read-only). Hệ thống không cung cấp chức năng Xóa (Delete) hay Sửa (Update) dữ liệu nhật ký cho bất kỳ ai, kể cả tài khoản Root Admin để đảm bảo tính minh bạch kiểm toán cao nhất. |
| Non-Functional Requirement | \- **NFR26-1:** Thao tác \"Ghi log\" vào cơ sở dữ liệu ở các Use Case khác phải được thiết kế chạy ngầm (Asynchronous) để không làm tăng thời gian phản hồi (Latency) của tiến trình chính. |

### Use case 27

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC27 |
| Use Case Name | Đăng nhập hệ thống |
| Description | Là một Người dùng, tôi muốn đăng nhập vào hệ thống bằng mật khẩu, mã OTP hoặc qua mạng xã hội (Google, Facebook) để truy cập vào phân quyền làm việc của mình. |
| Actor(s) | Mọi tác nhân (Khách hàng, Người bán, Quản trị viên), Hệ thống bên thứ 3 (Google, Facebook). |
| Priority | Must Have |
| Trigger | Người dùng truy cập trang Đăng nhập và lựa chọn phương thức đăng nhập mong muốn. |
| Pre-Condition(s) | Người dùng đã có tài khoản trên hệ thống hoặc có tài khoản mạng xã hội (Google/Facebook) đang hoạt động hợp lệ. |
| Post-Condition(s) | Đăng nhập thành công, hệ thống thiết lập phiên làm việc và điều hướng người dùng tới giao diện tương ứng. |
| Basic Flow | (Luồng chính: Đăng nhập bằng Email và Mật khẩu truyền thống) 1\. Người dùng chọn phương thức \"Đăng nhập bằng Mật khẩu\". 2\. Người dùng nhập Email và Mật khẩu đã đăng ký, sau đó nhấn \"Đăng nhập\". 3\. Hệ thống đối chiếu thông tin định danh trong cơ sở dữ liệu. 4\. Nếu hợp lệ, hệ thống thiết lập phiên làm việc cho người dùng. 5\. Hệ thống tự động nhận diện Phân quyền (Role) và điều hướng: Khách hàng về Trang chủ, Người bán/Admin về Bảng điều khiển. |
| Alternative Flow | 1a. Đăng nhập bằng mã OTP (Passwordless): 1a1. Người dùng nhập Email và chọn \"Đăng nhập bằng OTP\". 1a2. Hệ thống gửi mã xác thực 6 số qua Email người dùng. 1a3. Người dùng nhập mã OTP và xác nhận. 1a4. Hệ thống kiểm tra OTP hợp lệ. *Use Case tiếp tục ở Bước 4 của luồng chính.* 1b. Đăng nhập qua mạng xã hội (Google / Facebook): 1b1. Người dùng chọn nút \"Đăng nhập với Google\" (hoặc Facebook). 1b2. Hệ thống chuyển hướng sang màn hình xác thực của Google/Facebook. 1b3. Người dùng đồng ý cấp quyền truy cập thông tin cơ bản (Email, Tên). 1b4. Google/Facebook trả về thông báo xác thực thành công. (Nếu Email này chưa từng tồn tại, hệ thống tự động tạo mới tài khoản Khách hàng). *Use Case tiếp tục ở Bước 4 của luồng chính.* 1c. Đăng nhập bảo mật 2 lớp (Dành riêng cho Admin): 1c1. Quản trị viên truy cập đường dẫn riêng, nhập Email và Mật khẩu đúng. 1c2. Hệ thống yêu cầu xác thực bước 2 bằng cách gửi OTP về Email. 1c3. Admin nhập đúng OTP, hệ thống cấp phiên làm việc vào Admin Dashboard. |
| Exception Flow | 3a. Sai tài khoản hoặc mật khẩu: 3a1. Hệ thống báo lỗi \"Email hoặc mật khẩu không chính xác\" và chặn truy cập. *Use Case quay lại bước 2.* 1a3.1 (Ngoại lệ của 1a) Sai OTP: 1a3.2. Hệ thống báo lỗi \"Mã OTP không hợp lệ hoặc đã hết hạn\". 1b3.1 (Ngoại lệ của 1b) Từ chối cấp quyền: 1b3.2. Người dùng chọn lệnh \"Hủy\" trên màn hình của Google/Facebook. 1b3.3. Quá trình đăng nhập thất bại, hệ thống đưa người dùng về lại trang Đăng nhập mặc định. |
| Business Rules | \- **BR27-1:** Mã OTP chỉ có hiệu lực sử dụng 1 lần và tồn tại trong khoảng thời gian giới hạn (Ví dụ: 5 phút). \- **BR27-2:** Tính năng Đăng nhập qua mạng xã hội (Google/Facebook) mặc định chỉ cấp quyền Khách hàng (Customer). |
| Non-Functional Requirement | \- **NFR27-1:** Về mặt kỹ thuật, \"Phiên làm việc\" phải được quản lý bằng chuẩn mã hóa JSON Web Token (JWT) thông qua bộ lọc Spring Security. \- **NFR27-2:** Mật khẩu truyền thống của người dùng bắt buộc phải được mã hóa một chiều bằng BCrypt khi đối chiếu. \- **NFR27-3:** Tính năng đăng nhập Google/Facebook phải được giao tiếp thông qua giao thức chuẩn OAuth2. |

### Use case 28

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC28 |
| Use Case Name | Đăng xuất |
| Description | Là một Người dùng, tôi muốn đăng xuất khỏi hệ thống để kết thúc phiên làm việc an toàn, tránh bị người khác truy cập trái phép vào tài khoản của mình. |
| Actor(s) | Mọi tác nhân |
| Priority | Must Have |
| Trigger | Người dùng nhấn vào nút \"Đăng xuất\" (Logout) trên màn hình làm việc. |
| Pre-Condition(s) | Người dùng đang ở trạng thái đăng nhập hợp lệ. |
| Post-Condition(s) | Phiên làm việc kết thúc, hệ thống thu hồi quyền truy cập hiện tại của người dùng. |
| Basic Flow | 1\. Người dùng nhấp chọn lệnh \"Đăng xuất\". 2\. Hệ thống tiếp nhận yêu cầu và tiến hành hủy bỏ phiên làm việc hiện tại trên thiết bị của người dùng. 3\. Hệ thống xóa các dữ liệu cá nhân tạm thời đang hiển thị (như Giỏ hàng cá nhân, thông tin Hồ sơ). 4\. Hệ thống tự động điều hướng người dùng trở về Trang chủ mặc định ở trạng thái chưa đăng nhập. |
| Alternative Flow | *Không có. Luồng xử lý diễn ra trực tiếp một chiều.* |
| Exception Flow | *Không có.* |
| Business Rules | \- **BR28-1:** Ngay sau khi đăng xuất, mọi liên kết (URL) riêng tư mà người dùng cố tình truy cập lại (Ví dụ: Trang quản lý đơn, trang Checkout) đều phải bị hệ thống chặn lại và yêu cầu đăng nhập. |
| Non-Functional Requirement | \- **NFR28-1:** Thao tác đăng xuất phải được xử lý ngay lập tức tại phía giao diện (Client-side) bằng cách xóa JWT Token và làm sạch (Clear) Redux Store. |

### Use case 29

| Trường | **Nội dung** |
| --- | --- |
| Use Case ID | UC29 |
| Use Case Name | Đổi / Quên mật khẩu |
| Description | Là một Người dùng, tôi muốn thiết lập lại mật khẩu khi bị quên hoặc chủ động đổi mật khẩu để bảo vệ an toàn cho tài khoản cá nhân. |
| Actor(s) | Mọi tác nhân |
| Priority | Should Have |
| Trigger | Người dùng nhấn vào liên kết \"Quên mật khẩu\" ở màn hình Đăng nhập, hoặc chọn \"Đổi mật khẩu\" trong Quản lý tài khoản. |
| Pre-Condition(s) | Người dùng phải sở hữu (truy cập được) vào hòm thư Email đã đăng ký. |
| Post-Condition(s) | Mật khẩu mới được cập nhật thành công vào cơ sở dữ liệu. |
| Basic Flow | (Luồng Quên mật khẩu) 1\. Người dùng chọn lệnh \"Quên mật khẩu\" tại màn hình Đăng nhập. 2\. Hệ thống yêu cầu cung cấp Email định danh. 3\. Người dùng nhập Email và chọn lệnh \"Gửi mã xác thực\". 4\. Hệ thống tra cứu thông tin và gửi mã OTP xác nhận về hòm thư Email. 5\. Người dùng nhập mã OTP và nhập Mật khẩu mới mong muốn. 6\. Hệ thống xác thực OTP. Nếu hợp lệ, hệ thống tiến hành mã hóa bảo mật mật khẩu mới và ghi đè lên dữ liệu cũ. 7\. Hệ thống thông báo cập nhật thành công và đưa người dùng về lại trang Đăng nhập. |
| Alternative Flow | 1a. Luồng chủ động Đổi mật khẩu: 1a1. Người dùng đã đăng nhập, truy cập vào tab \"Đổi mật khẩu\" ở trang Hồ sơ. 1a2. Hệ thống yêu cầu nhập Mật khẩu hiện tại và Mật khẩu mới. 1a3. Hệ thống đối chiếu mật khẩu hiện tại. Nếu trùng khớp, hệ thống tiến hành cập nhật mật khẩu mới thành công. *Use Case kết thúc.* |
| Exception Flow | 4a. Email không tồn tại: 4a1. Hệ thống báo lỗi \"Tài khoản Email không tồn tại\" và chặn lệnh gửi mã. 1a3.1 (Ngoại lệ của 1a): Mật khẩu hiện tại không khớp: 1a3.2. Hệ thống báo lỗi bôi đỏ tại trường nhập liệu và từ chối cập nhật. |
| Business Rules | \- **BR29-1:** Mật khẩu bắt buộc phải có độ dài tối thiểu 8 ký tự để đảm bảo tiêu chuẩn an toàn. |
| Non-Functional Requirement | \- **NFR29-1:** Mật khẩu mới tuyệt đối không được lưu dưới dạng văn bản thô (Plain-text). Backend (Spring Boot) bắt buộc phải băm (hash) mật khẩu bằng thuật toán BCrypt trước khi lưu vào Database. |

