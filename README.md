**BỘ GIÁO DỤC VÀ ĐÀO TẠO**

**TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH**

**KHOA CÔNG NGHỆ THÔNG TIN**

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

![Trường Đại Học Sư Phạm Kỹ Thuật TP.
HCM](./image1.png){width="1.3031091426071741in"
height="1.6567169728783901in"}

**ĐỒ ÁN MÔN HỌC**

**LỚP HỌC PHẦN:**

**GVHD:**

**SINH VIÊN THỰC HIỆN: NHÓM 01**

  -----------------------------------------------------------------------
  Trác Ngọc Đăng Khoa                                    23110243
  ------------------------------------------------------ ----------------
  Nguyễn Thành Tin                                       23110343

  -----------------------------------------------------------------------

**Thành phố Hồ Chí Minh** -- **tháng năm 2026**

**BỘ GIÁO DỤC VÀ ĐÀO TẠO**

**TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH**

**KHOA CÔNG NGHỆ THÔNG TIN**

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

![Trường Đại Học Sư Phạm Kỹ Thuật TP.
HCM](./image1.png){width="1.3031091426071741in"
height="1.6567169728783901in"}

**ĐỒ ÁN MÔN HỌC**

**LỚP HỌC PHẦN:**

**GVHD:**

**SINH VIÊN THỰC HIỆN: NHÓM 01**

  -----------------------------------------------------------------------
  Trác Ngọc Đăng Khoa                                    23110243
  ------------------------------------------------------ ----------------
  Nguyễn Thành Tin                                       23110343

  -----------------------------------------------------------------------

**Thành phố Hồ Chí Minh** -- **tháng năm 2026**

**NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN**

Ngày , tháng 10 năm 2025

> GVHD

# MỤC LỤC {#mục-lục .unnumbered}

**DANH MỤC BẢNG**

**\
**

**DANH MỤC HÌNH ẢNH**

**\
**

**LỜI CẢM ƠN**

**PHẦN MỞ ĐẦU**

1.  **Lý do chọn đề tài**

2.  **Mục tiêu của đề tài**

3.  **Phạm vi nghiên cứu**

4.  **Phương pháp nghiên cứu**

# KHẢO SÁT HIỆN TRẠNG VÀ XÁC ĐỊNH YÊU CẦU

## Phân tích hiện trạng

Trong bối cảnh thương mại điện tử phát triển mạnh mẽ, việc xây dựng một
nền tảng bán hàng không chỉ dừng lại ở mô hình một cửa hàng mà đang
chuyển dịch sang mô hình sàn giao dịch thương mại điện tử đa nhà cung
cấp.

Hiện tại, người tiêu dùng cần một nền tảng tích hợp nơi họ có thể tìm
kiếm, lọc, sắp xếp sản phẩm từ nhiều nhà bán hàng khác nhau, quản lý giỏ
hàng mượt mà, thanh toán trực tuyến an toàn và tương tác qua Chatbot AI.
Về phía người bán, họ thiếu một công cụ tập trung để đăng bán sản phẩm,
theo dõi đơn hàng, cập nhật kho và thống kê doanh thu qua biểu đồ trực
quan. Về phía quản trị viên, cần có một hệ thống toàn diện để kiểm duyệt
người bán, quản lý mã giảm giá, các chương trình khuyến mãi và tùy biến
giao diện trang chủ động.

Việc xây dựng một hệ thống E-commerce đa nhà cung cấp là giải pháp hoàn
chỉnh, tự động hóa quy trình giao dịch, quản lý tài chính thông qua tích
hợp cổng thanh toán và nâng cao trải nghiệm người dùng

## Phân tích yêu cầu

### Yêu cầu chức năng

#### Yêu cầu chức năng nghiệp vụ

> **[Bảng yêu cầu chức năng nghiệp vụ]{.underline}**
>
> **Bộ phận: Khách vãng lai Mã số: GUEST**

  -------------------------------------------------------------------------------
  **STT**   **Công việc**       **Loại   **Quy         **Biểu   **Ghi chú**
                                công     định/Công     mẫu liên 
                                việc**   thức liên     quan**   
                                         quan**                 
  --------- ------------------- -------- ------------- -------- -----------------
  **1**     Xem giao diện Trang Tra cứu  QĐ_KVL1                Hiển thị các
            chủ                                                 Banner, Deals,
                                                                Lưới danh mục.

  **2**     Tìm kiếm, lọc sản   Tra cứu  QĐ_KVL2                Lọc theo giá,
            phẩm                                                màu, thương hiệu,
                                                                danh mục.

  **3**     Xem chi tiết sản    Tra cứu  QĐ_KVL3                Xem ảnh, giá, mô
            phẩm và Sản phẩm có                                 tả, tồn kho và
            liên quan                                           các gợi ý sản
                                                                phẩm cùng danh
                                                                mục.

  **4**     Xem các trang thông Tra cứu  QĐ_KVL4                Đọc tin tức từ hệ
            tin tĩnh (Xem bài                                   thống hoặc gian
            viết, tin tức, FAQ,                                 hàng. Xem FAQ,
            Chính sách giao                                     Chính sách giao
            hàng, Chính sách                                    hàng, Chính sách
            hoàn trả).                                          hoàn trả.

  **5**     Xem đánh giá        Tra cứu  QĐ_KVL5                Chỉ xem, không
            (Review)                                            được viết đánh
                                                                giá.

  **6**     Đăng ký / Đăng nhập Tương    QĐ_KVL6                Yêu cầu bắt buộc
                                tác                             để tiến hành mua
                                                                hàng.
  -------------------------------------------------------------------------------

Bảng 1‑1: Bảng yêu cầu chức năng nghiệp vụ Khách vãng lai

  -----------------------------------------------------------------------------
  **STT**   **Mã số**   **Tên Quy      **Mô tả chi tiết**             **Ghi
                        định/ Công                                    chú**
                        thức**                                        
  --------- ----------- -------------- ------------------------------ ---------
  **1**     QĐ_KVL1     Quy định hiển  Dữ liệu trang chủ (Homepage    API
                        thị Trang chủ  data) được hệ thống public mở  public
                                       hoàn toàn, Khách vãng lai      
                                       không cần truyền JWT Token vẫn 
                                       có thể tải được danh sách      
                                       Deals, Grid Category và các    
                                       sản phẩm nổi bật.              

  **2**     QĐ_KVL2     Quy định tìm   Hỗ trợ tìm kiếm bằng từ khóa.  
                        kiếm và lọc    Lọc nâng cao theo: Danh mục 3  
                                       cấp (Level 1, 2, 3), mức giá   
                                       (Min/Max), % giảm giá tối      
                                       thiểu, màu sắc và sắp xếp (Giá 
                                       từ thấp đến cao/Cao xuống      
                                       thấp).                         

  **3**     QĐ_KVL3     Quy định xem   Khách vãng lai được xem toàn   
                        chi tiết sản   bộ thông tin công khai của sản 
                        phẩm           phẩm bao gồm: Giá niêm yết     
                                       (MRP Price), Giá bán (Selling  
                                       Price), % giảm giá, hình ảnh   
                                       (từ Cloudinary), màu sắc và    
                                       thông tin Cửa hàng (Seller). Ở 
                                       cuối trang chi tiết, hệ thống  
                                       tự động hiển thị danh sách các 
                                       \"Sản phẩm liên quan\"         
                                       (Related products) dựa trên    
                                       cùng Category Level 3 để tăng  
                                       trải nghiệm mua sắm.           

  **4**     QĐ_KVL4     Quy định xem   Khách vãng lai có thể truy cập 
                        bài viết, tin  đọc các bài viết quảng bá, tin 
                        tức            tức sự kiện do Admin phát hành 
                                       trên toàn sàn hoặc bài viết    
                                       nội bộ của từng gian hàng      
                                       Seller, các trang thông tin    
                                       tĩnh (Static pages) như FAQ,   
                                       Điều khoản dịch vụ được public 
                                       hoàn toàn để khách hàng tìm    
                                       hiểu trước khi đăng ký.        

  **5**     QĐ_KVL5     Quy định giới  Khách vãng lai chỉ có quyền    
                        hạn đánh giá   ĐỌC các số sao (Rating) và nội 
                                       dung bình luận (Review) của    
                                       các sản phẩm. Tuyệt đối không  
                                       được phép GỬI đánh giá mới.    

  **6**     QĐ_KVL6     Quy định giới  Khách vãng lai **không có giỏ  
                        hạn nghiệp vụ  hàng (Cart)** và **không có    
                        Mua sắm        Danh sách yêu thích            
                                       (Wishlist)**. Nếu cố tình nhấn 
                                       nút \"Thêm vào giỏ\" hoặc      
                                       \"Mua ngay\", hệ thống (React  
                                       Router) bắt buộc chuyển hướng  
                                       (Redirect) sang trang Đăng     
                                       nhập / Đăng ký qua OTP.        
  -----------------------------------------------------------------------------

Bảng 1‑2: Bảng yêu quy định/ công thức liên quan Khách vãng lai

> **Bộ phận: Quản trị viên Mã số: ADMIN**

  -------------------------------------------------------------------------------
  **STT**   **Công việc**    **Loại     **Quy         **Biểu   **Ghi chú**
                             công       định/Công     mẫu liên 
                             việc**     thức liên     quan**   
                                        quan**                 
  --------- ---------------- ---------- ------------- -------- ------------------
  **1**     Quản lý kiểm     Tra        QĐ_AD1                 Thay đổi trạng
            duyệt Seller     cứu/Lưu                           thái tài khoản
                             trữ                               

  **2**     Quản lý danh mục Tra        QĐ_AD2                 Quản lý cấu trúc 3
            sản phẩm         cứu/Lưu                           cấp
                             trữ                               

  **3**     Quản lý trang    Lưu        QĐ_AD3                 Banner, lưới danh
            chủ (Homepage)   trữ/Cập                           mục
                             nhật                              

  **4**     Quản lý khuyến   Tra        QĐ_AD4                 Áp dụng toàn sàn
            mãi (Deals &     cứu/Lưu                           
            Coupons)         trữ                               

  **5**     Quản lý tài      Tra        QĐ_AD5                 
            khoản toàn hệ    cứu/Lưu                           
            thống            trữ                               

  **6**     Quản lý khiếu    Tra cứu/Xử QĐ_AD6                 Quyết định cuối
            nại (Disputes)   lý                                cùng giữa Khách và
                                                               Seller.

  **7**     Xử lý Hoàn tiền  Tính       QĐ_AD7                 
                             toán/Xử lý                        

  **8**     Cấu hình hệ      Lưu        QĐ_AD8                 Thiết lập tỉ lệ
            thống Xu (Coins) trữ/Cập                           quy đổi và hạn mức
                             nhật                              sử dụng Xu.

  **9**     Đối soát doanh   Tính       CT_AD1                 Quản lý dòng tiền
            thu Seller       toán/Kết                          và tính toán phí
                             xuất                              nền tảng.

  **10**    Quản lý đơn vị   Tra        QĐ_AD10                
            vận chuyển       cứu/Lưu                           
                             trữ                               

  **11**    Kiểm duyệt Sản   Tra cứu/Xử QĐ_AD11                Phê duyệt
            phẩm             lý                                (Approve) hoặc Từ
                                                               chối (Reject) sản
                                                               phẩm mới do Seller
                                                               đăng tải.

  **12**    Quản lý Cấu hình Cập nhật   QĐ_AD12                Giao diện cấu hình
            Phí nền tảng                                       % Platform fee thu
                                                               từ Seller.
  -------------------------------------------------------------------------------

Bảng 1‑1: Bảng yêu cầu chức năng nghiệp vụ Admin

+---+----------+-------------+-----------------------------+--------+
| * | **Mã     | **Tên Quy   | **Mô tả chi tiết**          | **Ghi  |
| * | số**     | định/ Công  |                             | chú**  |
| S |          | thức**      |                             |        |
| T |          |             |                             |        |
| T |          |             |                             |        |
| * |          |             |                             |        |
| * |          |             |                             |        |
+===+==========+=============+=============================+========+
| * | QĐ_AD1   | Quy định    | Khi Seller đăng ký (Cung    |        |
| * |          | kiểm duyệt  | cấp GST, thông tin ngân     |        |
| 1 |          | Seller      | hàng), tài khoản ở trạng    |        |
| * |          |             | thái PENDING. Admin kiểm    |        |
| * |          |             | duyệt và chuyển thành       |        |
|   |          |             | ACTIVE. Nếu vi phạm, Admin  |        |
|   |          |             | có thể SUSPEND hoặc BAN.    |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD2   | Quy định    | Danh mục được tổ chức theo  |        |
| * |          | quản lý     | 3 cấp độ (Level 1, Level 2, |        |
| 2 |          | danh mục    | Level 3).                   |        |
| * |          |             |                             |        |
| * |          |             |                             |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD3   | Quy định    | Admin được phép thiết lập   |        |
| * |          | quản lý     | lưới danh mục hiển thị,     |        |
| 3 |          | trang chủ   | danh mục đồ điện tử, nội    |        |
| * |          |             | thất và cập nhật các hình   |        |
| * |          |             | ảnh hiển thị trên trang     |        |
|   |          |             | chủ.                        |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD4   | Quy định    | Admin tạo Deal giảm giá cho |        |
| * |          | quản lý     | các danh mục hoặc phát hành |        |
| 4 |          | khuyến mãi  | Coupon chung (yêu cầu mã    |        |
| * |          |             | code, phần trăm giảm, thời  |        |
| * |          |             | hạn, giá trị tối thiểu).    |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD5   | Quy định    | Quản lý quyền truy cập và   |        |
| * |          | quản lý tài | thông tin của tất cả        |        |
| 5 |          | khoản       | Customer và Seller.         |        |
| * |          |             |                             |        |
| * |          |             |                             |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD6   | Quy định xử | Đối với các đơn hàng        |        |
| * |          | lý khiếu    | DISPUTED, Admin đóng vai    |        |
| 6 |          | nại         | trò trọng tài xem xét bằng  |        |
| * |          |             | chứng của cả Khách hàng và  |        |
| * |          |             | Seller. Phán quyết của      |        |
|   |          |             | Admin (Chấp nhận hoàn tiền  |        |
|   |          |             | hoặc Không chấp nhận) là    |        |
|   |          |             | quyết định cuối cùng.       |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD7   | Quy định    | Khi yêu cầu hoàn tiền được  |        |
| * |          | hoàn tiền   | duyệt (bởi Seller hoặc      |        |
| 7 |          |             | Admin), hệ thống Back-end   |        |
| * |          |             | tự động gọi API Refund của  |        |
| * |          |             | cổng thanh toán để trả tiền |        |
|   |          |             | về tài khoản ngân hàng gốc  |        |
|   |          |             | của khách hàng. Trạng thái  |        |
|   |          |             | đơn chuyển thành REFUNDED   |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD8   | Quy định    | Admin thiết lập 3 thông số  |        |
| * |          | cấu hình Xu | chính:                      |        |
| 8 |          |             |                             |        |
| * |          |             | 1\. **Tỉ lệ kiếm xu** (VD:  |        |
| * |          |             | 1000đ = 1 Xu).              |        |
|   |          |             |                             |        |
|   |          |             | 2\. **Tỉ giá tiêu xu** (VD: |        |
|   |          |             | 1 Xu = 1đ).                 |        |
|   |          |             |                             |        |
|   |          |             | 3\. **Hạn mức thanh toán**  |        |
|   |          |             | (VD: Xu chỉ thanh toán tối  |        |
|   |          |             | đa 50% giá trị đơn hàng).   |        |
+---+----------+-------------+-----------------------------+--------+
| * | CT_AD1   | Công thức   | Doanh thu Seller thực nhận  | Yêu    |
| * |          | đối soát    | = Tổng Selling Price - Phí  | cầu    |
| 9 |          | doanh thu   | nền tảng (Áp dụng theo %    | đối    |
| * |          | Seller      | được cấu hình tại QĐ_AD12). | soát   |
| * |          |             |                             | dòng   |
|   |          |             | *Lưu ý:* Phần tiền mà khách | tiền   |
|   |          |             | hàng đã dùng Xu để trừ      | minh   |
|   |          |             | thẳng vào đơn hàng sẽ do    | bạch.  |
|   |          |             | Admin bù lại vào ví của     |        |
|   |          |             | Seller trong quá trình đối  |        |
|   |          |             | soát để không làm thiệt hại |        |
|   |          |             | đến doanh thu của Seller.   |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD10  | Quy định    | Admin có quyền quản lý danh |        |
| * |          | Quản lý Đơn | sách các đối tác vận chuyển |        |
| 1 |          | vị Vận      | trên sàn. Cho phép Thêm     |        |
| 0 |          | chuyển      | mới, Cập nhật thông tin,    |        |
| * |          |             | hoặc Bật/Tắt                |        |
| * |          |             | (Active/Deactive) các ĐVVC. |        |
|   |          |             | Chỉ các ĐVVC ở trạng thái   |        |
|   |          |             | Active mới được phép hiển   |        |
|   |          |             | thị ở bước xử lý đơn hàng   |        |
|   |          |             | của Seller và Khách hàng.   |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD11  | Quy định    | Để bảo vệ nền tảng khỏi     |        |
| * |          | kiểm duyệt  | hàng giả/hàng cấm, các sản  |        |
| 1 |          | sản phẩm    | phẩm do Seller đăng tải sẽ  |        |
| 1 |          |             | ở trạng thái chờ duyệt.     |        |
| * |          |             | Admin xem xét thông tin và  |        |
| * |          |             | Approve (cho phép hiển thị) |        |
|   |          |             | hoặc Reject (yêu cầu sửa    |        |
|   |          |             | đổi).                       |        |
+---+----------+-------------+-----------------------------+--------+
| * | QĐ_AD12  | Quy định    | Admin có quyền thiết lập tỷ |        |
| * |          | thiết lập   | lệ Phần trăm Phí nền tảng   |        |
| 1 |          | phí sàn     | (Platform fee) áp dụng cho  |        |
| 2 |          |             | các giao dịch thành công    |        |
| * |          |             | thông qua giao diện cấu     |        |
| * |          |             | hình động, không fix cứng   |        |
|   |          |             | trong mã nguồn.             |        |
+---+----------+-------------+-----------------------------+--------+

Bảng 1‑2: Bảng yêu quy định/ công thức liên quan Admin

**Bộ phận: Nhà cung cấp (Seller) Mã số: SL**

  -----------------------------------------------------------------------------
  **STT**   **Công việc**   **Loại     **Quy         **Biểu   **Ghi chú**
                            công       định/Công     mẫu liên 
                            việc**     thức liên     quan**   
                                       quan**                 
  --------- --------------- ---------- ------------- -------- -----------------
  **1**     Cập nhật hồ sơ  Lưu trữ    QĐ_SL1                 Cung cấp GST,
            & thanh toán                                      Ngân hàng

  **2**     Quản lý sản     Lưu        QĐ_SL2                 Thêm biến thể,
            phẩm            trữ/Cập                           giá bán
                            nhật                              

  **3**     Xử lý đơn hàng  Cập nhật   QĐ_SL3                 Tự động hóa qua
                                                              API GHTK/Grab.

  **4**     In phiếu giao   Kết xuất   QĐ_SL4                 Mã vạch/QR code
            hàng (Vận đơn)                                    để dán lên gói
                                                              hàng.

  **5**     Báo cáo doanh   Kết xuất   CT_SL1                 Xem biểu đồ doanh
            thu & Thống kê                                    số

  **6**     Xử lý Yêu cầu   Tra cứu/Xử QĐ_SL5                 Phê duyệt hoặc từ
            trả hàng/Hoàn   lý                                chối yêu cầu từ
            tiền                                              khách.

  **7**     Cập nhật thống  Kết xuất   CT_SL2                 Tự động cập nhật
            kê hoàn tiền                                      vào Total Refund

  **8**     Chat trực tuyến Tương tác  QĐ_SL6                 Nhắn tin hỗ trợ
            (Real-time)                                       trực tiếp Khách
                                                              hàng.

  **9**     Xuất báo cáo    Kết xuất   QĐ_SL7                 Kết xuất dữ liệu
            Excel                                             đối soát ra file
                                                              .xlsx.
  -----------------------------------------------------------------------------

Bảng 1‑3: Bảng yêu cầu chức năng nghiệp vụ seller

+----+---------+---------------+------------------------------+--------+
| *  | **Mã    | **Tên Quy     | **Mô tả chi tiết**           | **Ghi  |
| *S | số**    | định/ Công    |                              | chú**  |
| TT |         | thức**        |                              |        |
| ** |         |               |                              |        |
+====+=========+===============+==============================+========+
| *  | QĐ_SL1  | Quy định cập  | Seller phải cung cấp mã số   |        |
| *1 |         | nhật hồ sơ    | thuế doanh nghiệp (GST), địa |        |
| ** |         |               | chỉ kho hàng (Pickup         |        |
|    |         |               | address), và thông tin ngân  |        |
|    |         |               | hàng hợp lệ để đối soát.     |        |
+----+---------+---------------+------------------------------+--------+
| *  | QĐ_SL2  | Quy định đăng | Mỗi sản phẩm phải thuộc 1    |        |
| *2 |         | tải sản phẩm  | danh mục Level 3. Bắt buộc   |        |
| ** |         |               | có Giá gốc (MRP) và Giá bán  |        |
|    |         |               | thực tế (Selling Price). Hệ  |        |
|    |         |               | thống tự động tính %         |        |
|    |         |               | Discount. Hình ảnh lưu qua   |        |
|    |         |               | Cloudinary.                  |        |
+----+---------+---------------+------------------------------+--------+
| *  | QĐ_SL3  | Quy định xử   | Khi đơn hàng có trạng thái   | Đồng   |
| *3 |         | lý đơn hàng & | CONFIRMED, Seller sử dụng    | bộ     |
| ** |         | Vận chuyển    | chức năng **\"Đẩy đơn vận    | Rea    |
|    |         |               | chuyển\"**. Hệ thống         | l-time |
|    |         |               | Back-end tự động gọi API của |        |
|    |         |               | ĐVVC (GHTK/Grab) để lấy Mã   |        |
|    |         |               | vận đơn. Các trạng thái tiếp |        |
|    |         |               | theo (SHIPPED, DELIVERED) sẽ |        |
|    |         |               | do **Webhook của ĐVVC tự     |        |
|    |         |               | động cập nhật về hệ thống**, |        |
|    |         |               | Seller không cần thao tác    |        |
|    |         |               | tay.                         |        |
+----+---------+---------------+------------------------------+--------+
| *  | CT_SL1  | Tính toán báo | Báo cáo tự động tổng hợp:    |        |
| *4 |         | cáo doanh thu | Tổng thu nhập (Total         |        |
| ** |         |               | Earning), Tổng số đơn (Total |        |
|    |         |               | Orders), Đơn bị hủy          |        |
|    |         |               | (Canceled Orders) và thể     |        |
|    |         |               | hiện qua biểu đồ trực quan.  |        |
+----+---------+---------------+------------------------------+--------+
| *  | **Q     | Quy định in   | Sau khi đẩy đơn thành công,  |        |
| *5 | Đ_SL4** | vận đơn       | Seller được phép kết xuất và |        |
| ** |         |               | in Phiếu giao hàng chứa Mã   |        |
|    |         |               | vận đơn (Tracking Code) định |        |
|    |         |               | dạng PDF để đóng gói.        |        |
+----+---------+---------------+------------------------------+--------+
| *  | QĐ_SL5  | Quy định xử   | Xử lý yêu cầu trả hàng \|    |        |
| *6 |         | lý yêu cầu    | Khi nhận được yêu cầu        |        |
| ** |         | trả hàng      | RETURN_REQUESTED, Seller có  |        |
|    |         |               | tối đa 3 ngày để phản hồi.   |        |
|    |         |               |                              |        |
|    |         |               | \- Nếu **Chấp nhận**: Chờ    |        |
|    |         |               | nhận lại hàng, sau đó xác    |        |
|    |         |               | nhận để hệ thống hoàn tiền.  |        |
|    |         |               |                              |        |
|    |         |               | \- Nếu **Từ chối**: Phải ghi |        |
|    |         |               | rõ lý do từ chối.            |        |
+----+---------+---------------+------------------------------+--------+
| *  | CT_SL2  | Cập nhật tổng | Khi một đơn hàng hoàn tiền   |        |
| *7 |         | hoàn tiền     | thành công, số tiền này bị   |        |
| ** |         | (Total        | trừ khỏi tổng thu nhập       |        |
|    |         | Refund).      | (Total Earning) và được cộng |        |
|    |         |               | dồn vào thống kê Total       |        |
|    |         |               | Refund trên Seller Dashboard |        |
+----+---------+---------------+------------------------------+--------+
| *  | QĐ_SL6  | Quy định hệ   | Hệ thống cung cấp công cụ    |        |
| *8 |         | thống chat    | Chat Real-time (sử dụng      |        |
| ** |         | trực tuyến    | WebSockets) giúp Seller giải |        |
|    |         |               | đáp thắc mắc của khách hàng  |        |
|    |         |               | về sản phẩm/đơn hàng ngay    |        |
|    |         |               | lập tức để tăng tỷ lệ chốt   |        |
|    |         |               | Sale.                        |        |
+----+---------+---------------+------------------------------+--------+
| *  | QĐ_SL7  | Quy định xuất | Cho phép Seller trích xuất   |        |
| *9 |         | báo cáo       | toàn bộ dữ liệu Lịch sử giao |        |
| ** |         |               | dịch (Transactions) và Doanh |        |
|    |         |               | thu ra tệp tin Excel (.xlsx) |        |
|    |         |               | để phục vụ nghiệp vụ đối     |        |
|    |         |               | soát và kế toán nội bộ.      |        |
+----+---------+---------------+------------------------------+--------+

Bảng 1‑4: Bảng yêu quy định/ công thức liên quan Nhân viên

**Bộ phận: Khách hàng (Customer)           Mã số: KH**

  -------------------------------------------------------------------------------
  **STT**   **Công       **Loại công **Quy        **Biểu   **Ghi chú**
            việc**       việc**      định/Công    mẫu liên 
                                     thức liên    quan**   
                                     quan**                
  --------- ------------ ----------- ------------ -------- ----------------------
  **1**     Xác thực tài Tương       QĐ_KH1                
            khoản bằng   tác/Xác                           
            OTP          thực                              

  **2**     Tương tác    Tra cứu     QĐ_KH2                Hỏi đáp tự động.
            Chatbot AI                                     

  **3**     Tìm kiếm,    Tra cứu     QĐ_KH3                Lọc theo giá, màu,
            lọc sản phẩm                                   danh mục.

  **4**     Quản lý giỏ  Tương       QĐ_KH4                
            hàng &       tác/Lưu trữ                       
            Wishlist                                       

  **5**     Đặt hàng &   Xử lý/Tính  QĐ_KH5                Thanh toán qua
            thanh toán   toán                              Stripe/Razorpay.

  **6**     Theo dõi &   Tra cứu     QĐ_KH6                Xem trạng thái đơn
            quản lý đơn                                    hàng.
            hàng                                           

  **7**     Đánh giá &   Kết xuất    QĐ_KH7                Rating 1-5 sao, kèm
            Phản hồi                                       ảnh thực tế.

  **8**     Nhận thông   Tra         QĐ_KH8                Hỗ trợ qua chat, email
            báo, ưu đãi, cứu/Tương                         hoặc hotline.
            hỗ trợ       tác                               

  **9**     Yêu cầu trả  Tương       QĐ_KH9                Chỉ áp dụng cho đơn đã
            hàng & hoàn  tác/Cập                           giao thành công.
            tiền         nhật                              

  **10**    Khiếu nại    Tương       QĐ_KH10               Dùng khi Seller từ
            lên Admin    tác/Xử lý                         chối yêu cầu trả hàng.
            (Dispute)                                      

  **11**    Quản lý ví   Tra cứu     QĐ_KH11               Xem số dư xu hiện tại
            Xu (Reward                                     và lịch sử nhận/tiêu
            Coins)                                         xu.

  **12**    Áp dụng Xu   Tính        QĐ_KH12               Trừ tiền tương ứng với
            khi thanh    toán/Xử lý                        số xu khách muốn sử
            toán                                           dụng.

  **13**    Chat trực    Tương tác   QĐ_KH13               Nhắn tin trực tiếp
            tuyến với                                      theo thời gian thực
            Người bán                                      (WebSockets) với
            (Real-time                                     Seller để nhận tư vấn
                                                           cụ thể về món hàng.

  **14**    Quản lý đa   Cập         QĐ_KH14               Thêm mới, cập nhật,
            địa chỉ giao nhật/Lưu                          xóa các địa chỉ trong
            hàng         trữ                               Sổ địa chỉ.
  -------------------------------------------------------------------------------

Bảng 1‑5: Bảng yêu cầu chức năng nghiệp vụ Khách hàng

+----+----------+-------------+---------------------------+-----------+
| *  | **Mã     | **Tên Quy   | **Mô tả chi tiết**        | **Ghi     |
| *S | số**     | định/ Công  |                           | chú**     |
| TT |          | thức**      |                           |           |
| ** |          |             |                           |           |
+====+==========+=============+===========================+===========+
| *  | QĐ_KH1   | Quy định    | Xác thực thông qua email  |           |
| *1 |          | đăng nhập   | sử dụng Java Mail Sender. |           |
| ** |          | bằng OTP    | Hệ thống gửi OTP gồm 6    |           |
|    |          |             | chữ số để đăng ký/đăng    |           |
|    |          |             | nhập, thời gian hiệu lực  |           |
|    |          |             | giới hạn.                 |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH2   | Quy định    | Chatbot có khả năng truy  |           |
| *2 |          | tương tác   | xuất cơ sở dữ liệu để trả |           |
| ** |          | Chatbot AI  | lời các câu hỏi về: Tình  |           |
|    |          |             | trạng đơn hàng, tổng tiền |           |
|    |          |             | giỏ hàng, thông tin chi   |           |
|    |          |             | tiết sản phẩm và khuyến   |           |
|    |          |             | mãi.                      |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH3   | Quy định    | Hỗ trợ tìm kiếm theo từ   |           |
| *3 |          | tìm kiếm và | khóa. Lọc nâng cao theo:  |           |
| ** |          | lọc         | Danh mục, mức giá         |           |
|    |          |             | (Min/Max), % giảm giá tối |           |
|    |          |             | thiểu, màu sắc và sắp xếp |           |
|    |          |             | (Giá từ thấp đến cao/Cao  |           |
|    |          |             | xuống thấp).              |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH4   | Quy định    | **Nghiệp vụ cốt lõi:**    | Tính toán |
| *4 |          | tách đơn    | Một giỏ hàng có thể chứa  | tổng      |
| ** |          | hàng giỏ    | sản phẩm từ nhiều Seller. | tiền:     |
|    |          | hàng        | Khi Checkout, hệ thống    | CT_KH1    |
|    |          |             | nhóm các món hàng theo    |           |
|    |          |             | Seller ID thành các       |           |
|    |          |             | Orders riêng biệt tương   |           |
|    |          |             | ứng với từng Seller,      |           |
|    |          |             | nhưng gộp chung vào 1     |           |
|    |          |             | PaymentOrder duy nhất để  |           |
|    |          |             | thanh toán 1 lần.         |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH5   | Quy định    | Hỗ trợ cổng thanh toán    |           |
| *5 |          | thanh toán  | VnPay, SePay hoặc Momo.   |           |
| ** |          | quốc tế/nội | Thanh toán thành công sẽ  |           |
|    |          | địa         | đổi trạng thái            |           |
|    |          |             | PaymentOrder thành        |           |
|    |          |             | SUCCESS và tự động trừ    |           |
|    |          |             | hàng trong kho.           |           |
+----+----------+-------------+---------------------------+-----------+
| 6  | QĐ_KH6   | Quy định    | Khách hàng tra cứu nhật   | Giao tiếp |
|    |          | theo dõi &  | ký vận chuyển **trực tiếp | qua API   |
|    |          | quản lý đơn | ngay trên giao diện       | ĐVVC      |
|    |          | hàng        | website** thông qua thanh |           |
|    |          |             | tiến trình (Order         |           |
|    |          |             | Stepper). Hệ thống liên   |           |
|    |          |             | tục đồng bộ và hiển thị   |           |
|    |          |             | chi tiết các mốc thời     |           |
|    |          |             | gian, vị trí và trạng     |           |
|    |          |             | thái giao hàng từ Đơn vị  |           |
|    |          |             | vận chuyển (GHTK/Grab).   |           |
|    |          |             | Khách hàng được cung cấp  |           |
|    |          |             | Mã vận đơn để đối chiếu   |           |
|    |          |             | nếu cần, nhưng không bắt  |           |
|    |          |             | buộc phải rời khỏi sàn để |           |
|    |          |             | tra cứu. Lịch sử đơn hàng |           |
|    |          |             | được lưu trữ tối thiểu 12 |           |
|    |          |             | tháng.                    |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH7   | Quy định    | Chỉ được đánh giá sản     |           |
| *7 |          | đánh giá    | phẩm sau khi đã nhận hàng |           |
| ** |          | (Review)    | (DELIVERED). Chấm điểm từ |           |
|    |          |             | 1 đến 5 sao, kèm bình     |           |
|    |          |             | luận và cho phép đính kèm |           |
|    |          |             | hình ảnh thực tế sản      |           |
|    |          |             | phẩm.                     |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH8   | Quy định    | Hệ thống gửi thông báo về |           |
| *8 |          | nhận thông  | đơn hàng, khuyến mãi, sự  |           |
| ** |          | báo, ưu     | kiện. Hỗ trợ khách hàng   |           |
|    |          | đãi, hỗ trợ | qua chat, email hoặc      |           |
|    |          |             | hotline.                  |           |
+----+----------+-------------+---------------------------+-----------+
| *  | QĐ_KH9   | Quy định    | Khách hàng chỉ được gửi   |           |
| *9 |          | yêu cầu trả | yêu cầu trả hàng/hoàn     |           |
| ** |          | hàng/hoàn   | tiền đối với đơn hàng có  |           |
|    |          | tiền        | trạng thái là DELIVERED   |           |
|    |          |             | (Đã giao) trong vòng **7  |           |
|    |          |             | ngày** kể từ ngày nhận.   |           |
|    |          |             | Bắt buộc phải cung cấp lý |           |
|    |          |             | do (hàng lỗi, sai         |           |
|    |          |             | mẫu\...) và đính kèm hình |           |
|    |          |             | ảnh/video minh chứng.     |           |
|    |          |             | Trạng thái đơn hàng       |           |
|    |          |             | chuyển sang               |           |
|    |          |             | RETURN_REQUESTED.         |           |
+----+----------+-------------+---------------------------+-----------+
| ** | QĐ_KH10  | Quy định    | Nếu Người bán từ chối yêu |           |
| 10 |          | khiếu nại   | cầu trả hàng, Khách hàng  |           |
| ** |          | (Escalate)  | có quyền nhấn nút \"Khiếu |           |
|    |          |             | nại lên Admin\". Đơn hàng |           |
|    |          |             | chuyển sang trạng thái    |           |
|    |          |             | DISPUTED (Đang tranh      |           |
|    |          |             | chấp) để Admin can thiệp. |           |
+----+----------+-------------+---------------------------+-----------+
| ** | QĐ_KH11  | Quy định    | Tích điểm (hoặc xu) cho   |           |
| 11 |          | Tích xu     | khách hàng dựa trên lịch  |           |
| ** |          |             | sử mua hàng. Khi đơn hàng |           |
|    |          |             | đạt trạng thái DELIVERED, |           |
|    |          |             | hệ thống tự động cộng số  |           |
|    |          |             | Xu = Final Payment Amount |           |
|    |          |             | \* Tỉ lệ tích xu (VD:     |           |
|    |          |             | 1%).                      |           |
+----+----------+-------------+---------------------------+-----------+
| ** | QĐ_KH12  | Quy định    | Khách hàng có thể dùng Xu |           |
| 12 |          | Tiêu xu     | ở bước Thanh toán. Số     |           |
| ** |          |             | tiền giảm được trừ trực   |           |
|    |          |             | tiếp vào Final Payment    |           |
|    |          |             | Amount. Có thể kết hợp sử |           |
|    |          |             | dụng Xu và Mã giảm giá    |           |
|    |          |             | (Coupon) cùng lúc.        |           |
+----+----------+-------------+---------------------------+-----------+
| ** | CT_KH1   | Công thức   | Tổng giá bán (Total       | Công thức |
| 13 |          | tính tiền   | Selling Price) = ∑ (Giá   | tổng quát |
| ** |          | Giỏ hàng    | bán × Số lượng)           | áp dụng   |
|    |          |             |                           | tại bước  |
|    |          |             | Giảm giá Coupon = Total   | Checkout, |
|    |          |             | Selling Price × (% Coupon | tự động   |
|    |          |             | / 100)                    | đối soát  |
|    |          |             |                           | cả Mã     |
|    |          |             | Giảm giá Xu = Số Xu sử    | giảm giá  |
|    |          |             | dụng × Tỉ giá             | và Xu     |
|    |          |             |                           | thưởng.   |
|    |          |             | Phí vận chuyển (Dynamic   |           |
|    |          |             | Shipping Fee): Được       |           |
|    |          |             | Back-end gọi API bên thứ  |           |
|    |          |             | 3 tính toán tự động dựa   |           |
|    |          |             | trên khoảng cách địa chỉ  |           |
|    |          |             | Kho Seller và Khách nhận. |           |
|    |          |             |                           |           |
|    |          |             | Tổng thanh toán (Final    |           |
|    |          |             | Payment) = Total Selling  |           |
|    |          |             | Price - Coupon - Xu + Phí |           |
|    |          |             | vận chuyển.               |           |
+----+----------+-------------+---------------------------+-----------+
| ** | QĐ_KH13  | Quy định    | Khách hàng được cung cấp  | Yêu cầu   |
| 14 |          | Chat trực   | khung Chat để trò chuyện  | tích hợp  |
| ** |          | tuyến với   | trực tiếp với Người bán   | We        |
|    |          | Seller      | (Seller) của sản phẩm đó. | bSockets. |
|    |          |             | Tin nhắn phải được cập    |           |
|    |          |             | nhật ngay lập tức         |           |
|    |          |             | (Real-time) ở cả 2 phía   |           |
|    |          |             | mà không cần tải lại      |           |
|    |          |             | trang                     |           |
+----+----------+-------------+---------------------------+-----------+
| ** | QĐ_KH14  | Quy định    | Khách hàng có quyền tạo   |           |
| 15 |          | quản lý địa | và quản lý nhiều địa chỉ  |           |
| ** |          | chỉ giao    | giao hàng khác nhau (Nhà  |           |
|    |          | hàng        | riêng, Cơ quan) trong Hồ  |           |
|    |          |             | sơ cá nhân. Tại bước      |           |
|    |          |             | Thanh toán (Checkout),    |           |
|    |          |             | khách hàng có thể chọn    |           |
|    |          |             | nhanh địa chỉ đã lưu hoặc |           |
|    |          |             | tạo địa chỉ mới.          |           |
+----+----------+-------------+---------------------------+-----------+

Bảng 1‑6: Bảng yêu quy định/ công thức liên quan Khách hàng

#### Yêu cầu chức năng hệ thống:

-   **Môi trường:** Hệ thống được xây dựng trên nền tảng Web
    Application, hoạt động qua mạng Internet, hỗ trợ truy cập trên máy
    tính và thiết bị di động. Front-end phát triển bằng React,
    TypeScript, Tailwind CSS, MUI và Redux Toolkit. Back-end sử dụng
    Java Spring Boot, MySQL Database. Hệ thống tích hợp trực tiếp với
    Cloudinary để lưu trữ phương tiện và các cổng thanh toán VnPay,
    SePay hoặc Momo.

-   **Phân quyền:** Hệ thống phân chia người dùng thành 3 nhóm quyền
    chính bằng Spring Security và JSON Web Token (JWT):

    -   *Khách hàng (ROLE_CUSTOMER):* Có quyền truy cập giao diện cửa
        hàng, tra cứu sản phẩm, tương tác Chatbot, quản lý giỏ hàng,
        theo dõi đơn hàng cá nhân, và để lại đánh giá. Bị chặn truy cập
        vào các API thuộc quyền quản lý.

    -   *Người bán (ROLE_SELLER):* Được cấp quyền vào bảng điều khiển
        (Seller Dashboard). Quản lý không gian bán hàng, đăng sản phẩm,
        theo dõi và xử lý các đơn hàng thuộc quyền sở hữu của mình, xem
        thống kê dòng tiền.

    -   *Quản trị viên (ROLE_ADMIN):* Nắm quyền cao nhất vào bảng điều
        khiển hệ thống (Admin Dashboard). Quản lý trạng thái mọi tài
        khoản Seller/Customer, thiết lập giao diện Home page, cấu hình
        Coupon/Deal toàn sàn.

> **[Bảng Yêu cầu Hệ thống]{.underline}**

  -----------------------------------------------------------------------------
  **STT**   **Nội dung**      **Mô tả chi tiết**              **Ghi chú**
  --------- ----------------- ------------------------------- -----------------
  **1**     Nền tảng hoạt     Ứng dụng Web nhiều lớp          Đảm bảo tương
            động              (Client-Server 3-tier) xây dựng thích Responsive
                              với React và Spring Boot. Hệ    tốt trên đa thiết
                              thống hoạt động qua mạng        bị nhờ Tailwind
                              Internet, cho phép truy cập     CSS và MUI.
                              thông qua trình duyệt web trên  
                              máy tính hoặc thiết bị di động. 

  **2**     Tích hợp bên thứ  Tích hợp VNPay, một số tài      Yêu cầu API Keys
            ba                khoản ngân hàng; Java Mail      bảo mật chặt chẽ.
                              Sender gửi mã OTP; Cloudinary   
                              lưu trữ tài nguyên hình ảnh.    

  **3**     Bảo mật và Phân   Quản lý luồng truy cập qua JWT  Ngăn chặn truy
            quyền             Token. Mật khẩu người dùng băm  cập chéo giữa các
                              qua BCrypt. Phân định rõ 3      Roles.
                              roles: Admin, Seller, Customer. 
  -----------------------------------------------------------------------------

### Yêu cầu phi chức năng

#### Liên quan đến người dùng (Khách vãng lai, Khách hàng, Người bán, Quản trị viên):

-   **Tính tiến hóa (Khả năng mở rộng):**

    -   Hệ thống phải có kiến trúc linh hoạt, cho phép Quản trị viên
        (Admin) dễ dàng tùy biến giao diện trang chủ, cấu hình lưới danh
        mục sản phẩm, banner quảng cáo và các chương trình khuyến mãi
        (Deals) để phù hợp với thị hiếu thay đổi của khách hàng,.

    -   Phải dễ dàng nâng cấp, mở rộng trong tương lai để đáp ứng quy mô
        số lượng lớn Người bán (Seller) tham gia vào sàn. Đồng thời, hệ
        thống được thiết kế sẵn sàng tích hợp thêm các dịch vụ giao hàng
        nội địa của bên thứ ba (như GHTK, GHN\...) hoặc mở rộng các cổng
        thanh toán điện tử quốc tế mới bên cạnh Stripe và Razorpay hiện
        tại,.

-   **Tính tiện dụng (Dễ sử dụng - UX/UI):**

    -   Giao diện của hệ thống phải trực quan, thân thiện và có thiết kế
        đáp ứng (Responsive) hoàn toàn nhờ sử dụng Tailwind CSS và
        Material UI (MUI), đảm bảo hiển thị và thao tác mượt mà trên
        nhiều loại thiết bị (máy tính, máy tính bảng, điện thoại di
        động),,.

    -   Các luồng thao tác cốt lõi của Khách hàng như: tìm kiếm, lọc sản
        phẩm đa tiêu chí (khoảng giá, màu sắc, % giảm giá), chọn biến
        thể kích cỡ, thao tác giỏ hàng và thanh toán gộp cho đa nhà cung
        cấp phải diễn ra đơn giản và tiện lợi nhất,,.

    -   Hệ thống Bảng điều khiển (Dashboard) dành riêng cho Người bán và
        Quản trị viên phải được sắp xếp logic, các công cụ thống kê
        doanh thu thể hiện qua biểu đồ trực quan giúp giảm thiểu số lần
        nhấp chuột không cần thiết và nâng cao hiệu suất làm việc,,.

-   **Tính hiệu quả (Hiệu suất và Độ ổn định):**

    -   Nền tảng phải tối ưu hóa tốc độ tải trang (dưới 2 giây), phản
        hồi các thao tác tìm kiếm, chuyển đổi tab và phân trang
        (Pagination) nhanh chóng. Việc xử lý tải và lưu trữ khối lượng
        lớn hình ảnh sản phẩm đa phương tiện phải được thực hiện hiệu
        quả thông qua dịch vụ đám mây Cloudinary,.

    -   Hệ thống phải hoạt động ổn định và xử lý chính xác dòng tiền đối
        soát, áp dụng đúng mã giảm giá (Coupon)/Xu thưởng cho hàng loạt
        giao dịch thanh toán trực tuyến diễn ra cùng lúc, đặc biệt trong
        các khung giờ cao điểm có lưu lượng truy cập khổng lồ,.

-   **Tính tương thích:**

    -   Hệ thống web hoạt động và hiển thị nhất quán trên các trình
        duyệt hiện đại phổ biến (Google Chrome, Safari, Firefox, Edge).

    -   Phải đảm bảo tính đồng bộ dữ liệu theo thời gian thực
        (Real-time) giữa hoạt động đặt hàng của Khách hàng, tình trạng
        kho hàng và thông báo trạng thái vận chuyển hiển thị trên Bảng
        điều khiển của Người bán (Seller Dashboard)

#### Liên quan đến chuyên viên tin học (Đội ngũ phát triển):

-   **Tính tái sử dụng:**

    -   Ứng dụng Front-end (React) cần được xây dựng theo kiến trúc
        Component (ví dụ: tái sử dụng các component ProductCard,
        OrderTable, AddressForm, DrawerList ở nhiều màn hình khác
        nhau),.

    -   Cấu trúc Back-end (Spring Boot RESTful APIs) cần được tách biệt
        độc lập,, tuân thủ các chuẩn lập trình API giúp hệ thống dễ dàng
        được tái sử dụng để giao tiếp khi phát triển thêm nền tảng Ứng
        dụng di động (Mobile App) sau này.

-   **Tính bảo trì:**

    -   Mã nguồn dự án và cơ sở dữ liệu MySQL phải được phân tách theo
        các miền nghiệp vụ rõ ràng (Module Sản phẩm, Module Đơn hàng,
        Module Thanh toán, Module Tài khoản\...),,. Cấu trúc này cho
        phép đội ngũ bảo trì dễ dàng sửa lỗi hoặc mở rộng tính năng mới
        ở một phân hệ mà không gây đổ vỡ (Crash) tới các module khác của
        hệ thống.

-   **Tính bảo mật:**

    -   Mật khẩu của người dùng bắt buộc phải được mã hóa một chiều an
        toàn bằng thuật toán BCrypt trước khi lưu trữ vào Cơ sở dữ
        liệu,.

    -   Hệ thống phải áp dụng cơ chế xác thực phiên làm việc chặt chẽ
        bằng chuẩn JSON Web Token (JWT) thông qua bộ lọc Spring
        Security,,.

    -   Cần thiết lập kiểm soát truy cập dựa trên vai trò (Role-Based
        Access Control) để ngăn chặn tuyệt đối tình trạng truy cập chéo
        tài nguyên giữa 3 nhóm quyền biệt lập: Khách hàng
        (ROLE_CUSTOMER), Người bán (ROLE_SELLER) và Quản trị viên
        (ROLE_ADMIN),,. Thông tin nhạy cảm về thẻ tín dụng khi thanh
        toán phải được tuân thủ chuẩn bảo mật trực tiếp thông qua API
        của VnPay, SePay hoặc Momo.

> **Bảng Yêu cầu chất lượng**

  ----------------------------------------------------------------------------------
  **STT**   **Nội dung**      **Tiêu    **Mô tả chi tiết**       **Ghi chú**
                              chuẩn**                            
  --------- ----------------- --------- ------------------------ -------------------
  **1**     Xử lý đa giao     Hiệu quả  Xử lý tự động và chính   Nghiệp vụ cốt lõi
            dịch & Tách đơn   / Chính   xác việc tách giỏ hàng   của sàn
            hàng              xác       thành nhiều đơn hàng phụ Multivendor. Xử lý
                                        (tương ứng với từng      logic tại tầng
                                        Seller) và gộp thanh     Back-end (Spring
                                        toán một lần.            Boot).

  **2**     Tùy biến Trang    Tính tiến Admin dễ dàng tùy biến   Đáp ứng nhu cầu
            chủ & Khả năng mở hóa (Khả  linh hoạt banner, lưới   thay đổi giao diện
            rộng              năng mở   danh mục và khuyến mãi   theo các chiến dịch
                              rộng)     từ Dashboard. Hệ thống   Marketing mà không
                                        sẵn sàng mở rộng không   cần sửa code.
                                        giới hạn số lượng Seller 
                                        và tích hợp thêm cổng    
                                        thanh toán mới.          

  **3**     Trải nghiệm mua   Tính tiện Giao diện thân thiện,    Sử dụng Tailwind
            sắm UX/UI đa      dụng      hiển thị Responsive hoàn CSS và Material UI
            thiết bị                    hảo trên Mobile, Tablet, (MUI). Giảm tỷ lệ
                                        PC. Các thao tác tìm     thoát trang.
                                        kiếm, lọc sản phẩm đa    
                                        tiêu chí, đánh giá và    
                                        Checkout diễn ra mượt    
                                        mà, trực quan.           

  **4**     Tốc độ tải trang  Tính hiệu Thời gian tải trang      Tối ưu hóa truy vấn
            & Xử lý tải cao   quả (Hiệu (Homepage, Product       MySQL và lưu trữ
                              năng)     details) dưới 2 giây. Hệ media qua
                                        thống hoạt động ổn định  Cloudinary.
                                        khi có lưu lượng truy    
                                        cập đột biến (Flash      
                                        Sale, Deals).            

  **5**     Hiển thị đa trình Tính      Hệ thống Front-end tương Tối ưu trải nghiệm
            duyệt & Đồng bộ   tương     thích hoàn toàn với      liền mạch giữa
            dữ liệu           thích     Chrome, Safari, Firefox. người mua và người
                                        Đồng bộ trạng thái đơn   bán.
                                        hàng (từ Khách hàng tới  
                                        Seller Dashboard) theo   
                                        thời gian thực.          

  **6**     Kiến trúc API &   Tính tái  Các RESTful API (Spring  Tiết kiệm chi phí
            Component         sử dụng   Boot) và giao diện       và thời gian phát
                                        Front-end (React         triển mở rộng.
                                        Components) được xây     
                                        dựng độc lập. Dễ dàng    
                                        tái sử dụng API nếu phát 
                                        triển thêm Mobile App.   

  **7**     Module hóa & Sửa  Tính bảo  Mã nguồn phân tách rõ    Đảm bảo tính bền
            lỗi hệ thống      trì       ràng theo các miền       vững của dự án phần
                                        nghiệp vụ (Sản phẩm, Đơn mềm.
                                        hàng, User, Thanh toán). 
                                        Dễ dàng dò tìm lỗi, nâng 
                                        cấp tính năng mà không   
                                        gây \"crash\" chéo.      

  **8**     Xác thực & Mã hóa Tính bảo  Mật khẩu băm một chiều   Tích hợp Spring
            dữ liệu người     mật       (BCrypt). Luồng truy cập Security. Không lưu
            dùng                        kiểm soát chặt chẽ bằng  trữ thông tin thẻ
                                        JWT Token. Ngăn chặn     tín dụng nhạy cảm
                                        tuyệt đối việc truy cập  (Xử lý qua
                                        chéo tài nguyên giữa     Stripe/Razorpay).
                                        Admin, Seller và         
                                        Customer.                
  ----------------------------------------------------------------------------------

## Quy trình tác nghiệp

### Quy trình tham quan và chuyển đổi của Khách vãng lai

Quy trình trải nghiệm và chuyển đổi của một Khách vãng lai trên sàn
E-commerce diễn ra theo trình tự sau: Đầu tiên, Khách vãng lai truy cập
vào nền tảng thông qua các trình duyệt web (có thể từ link chia sẻ, tìm
kiếm Google hoặc trực tiếp URL). Hệ thống lập tức hiển thị **Trang chủ
(Homepage)** với các Banner quảng cáo, các chương trình Khuyến mãi
(Deals) đang diễn ra và các Danh mục nổi bật (Điện tử, Nội thất, Thời
trang\...) mà không yêu cầu đăng nhập.

Tiếp theo, khách tự do điều hướng, sử dụng thanh tìm kiếm (Search) hoặc
nhấp vào cây danh mục 3 cấp để duyệt sản phẩm. Tại trang danh sách,
khách sử dụng bộ lọc nâng cao (lọc theo khoảng giá, màu sắc, thương
hiệu) để thu hẹp kết quả. Khi tìm thấy sản phẩm ưng ý, khách nhấp vào để
xem **Chi tiết sản phẩm** (đọc mô tả, xem hình ảnh thực tế, xem đánh giá
1-5 sao từ người dùng trước) hoặc đọc các bài viết/tin tức liên quan đến
gian hàng đó để tăng độ tin cậy.

Khi khách quyết định mua hàng và thực hiện hành động nhấn nút **\"Thêm
vào giỏ hàng\" (Add to Cart)**, **\"Thêm vào Wishlist\"**, hoặc **\"Chat
với AI Chatbot\"**, hệ thống Spring Security ở Backend và React Router ở
Frontend sẽ chặn thao tác này và tự động bật Pop-up / chuyển hướng
(Redirect) khách sang **Màn hình Đăng nhập / Đăng ký**. Tại đây, khách
vãng lai bắt buộc phải nhập Email/SĐT và xác thực mã OTP. Sau khi nhập
OTP thành công, hệ thống cấp JWT Token, Khách vãng lai chính thức chuyển
đổi trạng thái thành **Khách hàng (Customer)** và được tiếp tục quy
trình mua sắm, thanh toán bị gián đoạn trước đó.

### Quy trình khách hàng mua sắm trực tuyến

Quy trình khách hàng mua sắm diễn ra theo các bước sau: Đầu tiên, khách
hàng truy cập vào nền tảng và tìm kiếm sản phẩm thông qua thanh tìm
kiếm, bộ lọc nâng cao (theo mức giá, màu sắc, % giảm giá) hoặc nhận tư
vấn trực tiếp từ AI Chatbot. Khi chọn được sản phẩm ưng ý, khách chọn
biến thể (size, màu sắc) và thêm vào giỏ hàng hoặc đưa vào danh sách yêu
thích (Wishlist) để mua sau.

Tại bước thanh toán, do đặc thù đa nhà cung cấp, hệ thống sẽ tự động
tách giỏ hàng thành các đơn hàng phụ tương ứng với từng người bán. Khách
hàng tiến hành nhập mã giảm giá (nếu có) có thể tích chọn sử dụng \"Xu
tích lũy\" từ ví tài khoản để trừ trực tiếp vào tổng số tiền phải trả.
Hệ thống sẽ tự động tính toán lại số tiền cuối cùng. Thanh toán một lần
duy nhất thông qua các cổng thanh toán trực tuyến an toàn như VnPay,
SePay hoặc Momo. Cuối cùng, hệ thống ghi nhận giao dịch thành công và
chuyển thông tin đơn hàng đến các gian hàng tương ứng.

### Quy trình khách hàng tra cứu đơn hàng và tương tác 

Để theo dõi đơn hàng, khách hàng đăng nhập vào hệ thống và chọn chức
năng "Đơn hàng của tôi" (My Orders). Tại đây, họ có thể xem trạng thái
hiện tại của đơn hàng, bao gồm các bước: Đã đặt (Placed), Đã xác nhận
(Confirmed), Đang giao (Shipped) và Đã giao (Delivered). Điểm đặc biệt
của hệ thống là khách hàng có thể mở giao diện AI Chatbot và hỏi trực
tiếp bằng ngôn ngữ tự nhiên (ví dụ: \"Tôi có bao nhiêu đơn hàng đã
giao?\") để tra cứu trạng thái đơn hàng hoặc chi tiết giỏ hàng nhanh
chóng.

Khi đơn hàng hoàn tất, hệ thống tự động kích hoạt tiến trình cộng Xu
thưởng vào ví của khách hàng dựa trên tổng giá trị thanh toán của đơn
hàng đó. Khách hàng có thể kiểm tra biến động số dư Xu tại màn hình Quản
lý tài khoản cá nhân. Khách hàng có thể đánh giá (từ 1-5 sao) và đính
kèm hình ảnh thực tế của sản phẩm. Nếu không vừa ý với món hàng, khách
hàng có thể yêu cầu hoàn tiền, trả hàng với lý do hợp lý và tuân thủ
đúng chính sách của sàn.

### Quy trình quản lý gian hàng và sản phẩm (Dành cho Người bán) 

Người bán (Seller) sau khi được cấp tài khoản sẽ đăng nhập vào hệ thống
bảng điều khiển riêng (Seller Dashboard). Họ thực hiện các thao tác quản
lý kho hàng bao gồm: thêm mới, chỉnh sửa hoặc xóa sản phẩm. Các thông
tin cần cung cấp gồm có tên, mô tả, giá gốc (MRP), giá bán thực tế, số
lượng tồn kho và tải hình ảnh lên hệ thống (thông qua Cloudinary). Khi
có sự thay đổi về giá gốc và giá bán, hệ thống tự động tính toán phần
trăm giảm giá để hiển thị. Ngoài ra, người bán cũng có trách nhiệm tiếp
nhận đơn hàng từ khách và cập nhật trạng thái xử lý đơn (từ Chờ xử lý
đến Đã giao hàng).

### Quy trình kiểm duyệt và quản trị nền tảng (Dành cho Admin) 

Quản trị viên (Admin) nắm quyền kiểm soát toàn bộ nền tảng thông qua
Bảng điều khiển quản trị. Khi một người bán mới đăng ký, tài khoản sẽ ở
trạng thái chờ duyệt (Pending Verification). Admin sẽ kiểm tra hồ sơ và
thực hiện phê duyệt (Active), hoặc có thể đình chỉ (Suspend), cấm vĩnh
viễn (Ban) đối với các tài khoản vi phạm chính sách. Bên cạnh quản lý
người dùng, Admin thực hiện việc tùy chỉnh giao diện trang chủ, thay đổi
lưới danh mục, banner, và phát hành các mã giảm giá (Coupon), chương
trình khuyến mãi (Deals) cho toàn bộ hệ thống.

### Quy trình thống kê và đối soát doanh thu 

Hệ thống tự động tổng hợp và tính toán các chỉ số kinh doanh theo thời
gian thực. Từ bảng điều khiển, Người bán có thể xem chi tiết tổng thu
nhập, tổng số sản phẩm đã bán, số lượng đơn hàng bị hủy và theo dõi lịch
sử dòng tiền (Transactions). Báo cáo doanh thu được xuất ra dưới dạng
các biểu đồ trực quan (Earning graphs) theo ngày, tuần hoặc tháng, giúp
người bán dễ dàng phân tích tình hình kinh doanh của gian hàng. Đồng
thời, dữ liệu này là cơ sở để hệ thống tiến hành đối soát và thanh toán
tiền hàng cho Người bán sau khi đơn hàng giao thành công.

### Quy trình yêu cầu trả hàng và hoàn tiền (Refund & Return Process) 

Quy trình xử lý trả hàng và hoàn tiền được diễn ra chặt chẽ giữa 3 bên
nhằm đảm bảo tính công bằng:

-   **Bước 1: Khởi tạo yêu cầu (Khách hàng):** Khách hàng đăng nhập,
    truy cập lịch sử mua hàng và chọn đơn hàng có trạng thái \"Đã giao\"
    (Delivered) trong thời hạn cho phép (VD: 7 ngày). Khách hàng chọn
    chức năng \"Yêu cầu trả hàng\", điền lý do và tải lên hình ảnh/video
    minh chứng. Hệ thống chuyển trạng thái đơn sang \"Yêu cầu trả hàng\"
    (Return Requested) và tạm thời đóng băng khoản tiền đối soát của đơn
    hàng này đối với Người bán.

-   **Bước 2: Xử lý yêu cầu (Người bán):** Người bán nhận được thông báo
    trên Seller Dashboard. Xem xét minh chứng của khách hàng.

    -   *Trường hợp 2a (Đồng ý):* Người bán bấm \"Chấp nhận\". Khách
        hàng gửi trả lại hàng. Khi Người bán nhận được hàng sẽ bấm \"Xác
        nhận hoàn tiền\". Trạng thái đơn chuyển thành \"Đã hoàn tiền\"
        (Refunded).

    -   *Trường hợp 2b (Từ chối):* Người bán bấm \"Từ chối\" kèm theo lý
        do.

-   **Bước 3: Khiếu nại (Khách hàng):** Nếu bị Người bán từ chối, Khách
    hàng có quyền nhấn \"Khiếu nại\". Đơn hàng chuyển sang trạng thái
    \"Tranh chấp\" (Disputed).

-   **Bước 4: Phán quyết (Quản trị viên - Admin):** Quản trị viên can
    thiệp vào các đơn \"Disputed\", kiểm tra đối chứng dữ liệu từ cả hai
    bên. Admin đưa ra phán quyết cuối cùng. Nếu Admin duyệt hoàn tiền,
    hệ thống sẽ tự động kích hoạt API của cổng thanh toán để đẩy tiền về
    thẻ của khách, đồng thời hệ thống tự động cập nhật biểu đồ thống kê
    \"Total Refund\" (Tổng số tiền hoàn) trên Dashboard của Người bán

# MÔ HÌNH HÓA YÊU CẦU

## Nhận diện tác nhân và chức năng trong sơ đồ Use case

Các Usecase đang được thiết kế khóa tổng quát, như các usecase Quản lý
bao gồm tra cứu, thêm, xóa, sửa. Có thể tách usecase ra riêng nhưng rất
dài

  ---------------------------------------------------------------------------
  **Tác nhân     **Mã UC**  **Tên Use Case   **Mô tả**
  (Actor)**                 (User Goal)**    
  -------------- ---------- ---------------- --------------------------------
  **Khách vãng   **UC01**   **Khám phá nền   Xem trang chủ (Banner, Deals),
  lai (Guest)**             tảng**           tìm kiếm, lọc sản phẩm nâng cao.
                                             Xem chi tiết sản phẩm, gợi ý sản
                                             phẩm liên quan và đọc các trang
                                             thông tin tĩnh (FAQ, Chính
                                             sách).

                 **UC02**   **Đăng ký tài    Khách vãng lai đăng ký tài khoản
                            khoản**          qua Email và xác thực bằng mã
                                             OTP để trở thành Khách hàng
                                             chính thức.

  **Khách hàng   **UC03**   **Quản lý Giỏ    Thêm sản phẩm vào giỏ, cập nhật
  (Customer)**              hàng (Cart)**    số lượng hoặc xóa sản phẩm. Tự
                                             động tính toán lại tổng tiền và
                                             % giảm giá.

                 **UC04**   **Quản lý Danh   Thêm các sản phẩm ưng ý vào
                            sách yêu thích** Wishlist để lưu trữ cho các lần
                                             mua sắm sau và xóa sản phẩm khỏi
                                             danh sách.

                 **UC05**   **Đặt hàng và    Chọn địa chỉ giao hàng, hệ thống
                            Thanh toán**     tự động tách đơn theo Seller
                                             (Split Order), áp dụng
                                             Coupon/Xu, và thanh toán qua
                                             cổng điện tử.

                 **UC06**   **Theo dõi &     Xem lịch sử mua hàng, tra cứu
                            Quản lý đơn      tiến trình vận chuyển theo thời
                            hàng**           gian thực và hủy đơn khi còn ở
                                             trạng thái \"Mới đặt\".

                 **UC07**   **Tương tác      Nhắn tin hỏi đáp tự động bằng
                            Chatbot AI**     ngôn ngữ tự nhiên để tra cứu
                                             thông tin sản phẩm, đơn hàng, và
                                             giỏ hàng.

                 **UC08**   **Đánh giá sản   Chấm điểm (1-5 sao), viết bình
                            phẩm**           luận và đính kèm hình ảnh thực
                                             tế sau khi đơn hàng đã \"Đã
                                             giao\".

                 **UC09**   **Yêu cầu Trả    Tạo yêu cầu đổi/trả hàng kèm
                            hàng & Hoàn      minh chứng. Có quyền Khiếu nại
                            tiền**           (Dispute) lên Admin nếu bị
                                             Seller từ chối.

                 **UC10**   **Cập nhật thông Khách hàng xem và chỉnh sửa các
                            tin cá nhân**    thông tin liên lạc cơ bản (Họ
                                             tên, Số điện thoại) để hệ thống
                                             cập nhật hồ sơ người dùng.

                 **UC11**   **Quản lý Địa    Khách hàng thực hiện thêm mới,
                            chỉ**            chỉnh sửa hoặc xóa các địa chỉ
                                             giao nhận hàng hóa (nhà riêng,
                                             cơ quan) lưu trong sổ địa chỉ.

                 **UC12**   **Theo dõi ví    Khách hàng tra cứu tổng số dư Ví
                            xu**             xu hiện tại và xem lại nhật ký
                                             chi tiết các lần được cộng/trừ
                                             xu qua từng đơn hàng.

  **Khách hàng & **UC13**   **Chat trực      Luồng giao tiếp dùng chung kết
  Người bán**               tuyến**          nối người mua và người bán thông
                                             qua kiến trúc WebSockets để trao
                                             đổi, tư vấn trực tiếp về sản
                                             phẩm/đơn hàng.

  **Người bán    **UC14**   **Quản lý Hồ sơ  Cập nhật thông tin doanh nghiệp
  (Seller)**                và Gian hàng**   (GST), tài khoản ngân hàng đối
                                             soát, địa chỉ kho lấy hàng và
                                             trang trí Banner gian hàng.

                 **UC15**   **Quản lý Kho    Đăng tải sản phẩm mới (chờ
                            sản phẩm**       duyệt), tải ảnh qua Cloudinary,
                                             cập nhật giá bán/tồn kho và cấu
                                             hình màu sắc/kích cỡ.

                 **UC16**   **Xử lý Đơn hàng Tiếp nhận đơn, xác nhận, tự động
                            & Vận chuyển**   đẩy đơn sang API hãng vận chuyển
                                             (GHTK/Grab) và kết xuất in Phiếu
                                             giao hàng.

                 **UC17**   **Xử lý Yêu cầu  Xem xét lý do và minh chứng từ
                            Hoàn trả**       Khách hàng để đưa ra quyết định
                                             Chấp nhận (cho phép hoàn tiền)
                                             hoặc Từ chối.

                 **UC18**   **Theo dõi Đối   Xem biểu đồ doanh thu tổng quan,
                            soát & Doanh     theo dõi dòng tiền đối soát và
                            thu**            thực hiện Xuất báo cáo dữ liệu
                                             ra file Excel.

  **Quản trị     **UC19**   **Kiểm duyệt     Xét duyệt hồ sơ đăng ký gian
  viên (Admin)**            Người bán**      hàng (Active), hoặc tạm đình chỉ
                                             (Suspend), cấm vĩnh viễn (Ban)
                                             tài khoản vi phạm.

                 **UC20**   **Kiểm duyệt Sản Xem xét thông tin các sản phẩm
                            phẩm**           mới do Seller đăng tải để Phê
                                             duyệt (cho phép hiển thị công
                                             khai) hoặc Từ chối.

                 **UC21**   **Quản lý Giao   Tùy biến linh hoạt giao diện
                            diện Trang chủ** trang chủ, cập nhật Banner, cấu
                                             hình lưới danh mục nổi bật mà
                                             không cần can thiệp code.

                 **UC22**   **Quản lý Chiến  Phát hành, cập nhật hoặc xóa các
                            dịch Khuyến      chiến dịch Marketing chung toàn
                            mãi**            sàn bao gồm Khuyến mãi (Deals)
                                             và Mã giảm giá (Coupons).

                 **UC23**   **Giải quyết     Xem xét các đơn hàng có tranh
                            khiếu nại        chấp, đưa ra phán quyết cuối
                            (Disputes)**     cùng và tự động gọi API hoàn
                                             tiền cho Khách hàng.

                 **UC24**   **Quản lý Khách  Xem danh sách khách hàng, theo
                            hàng**           dõi thông tin tài khoản và thực
                                             hiện khóa (Ban) các tài khoản vi
                                             phạm chính sách.

                 **UC25**   **Cấu hình Thông Thiết lập các thông số Ví Xu (tỉ
                            số Tài chính**   lệ quy đổi, hạn mức tiêu) và cấu
                                             hình % Phí nền tảng (Platform
                                             fee) áp dụng cho gian hàng.

                 **UC26**   **Theo dõi Nhật  Truy xuất và xem xét lịch sử các
                            ký**             thao tác thay đổi dữ liệu quan
                                             trọng trên hệ thống (kiểm duyệt,
                                             xóa dữ liệu) để kiểm toán.

  **Mọi Tác      **UC27**   **Đăng nhập**    Người dùng xác thực danh tính
  nhân**                                     qua Email/Mật khẩu (đối với
                                             Admin) hoặc đăng nhập không mật
                                             khẩu qua mã OTP (đối với Khách
                                             hàng, Người bán). Hệ thống cấp
                                             phiên làm việc và điều hướng
                                             theo phân quyền.

                 **\        **Đăng xuất**    Người dùng chủ động kết thúc
                 UC28**                      phiên làm việc. Hệ thống tiến
                                             hành hủy xóa khỏi Local Storage
                                             thiết bị và điều hướng người
                                             dùng về trang chủ mặt tiền một
                                             cách an toàn.

                 **UC29**   **Đổi / Quên mật Hỗ trợ người dùng yêu cầu thiết
                            khẩu**           lập lại mật khẩu khi quên hoặc
                                             chủ động đổi mật khẩu để bảo vệ
                                             tài khoản. Quá trình này bắt
                                             buộc phải xác thực bảo mật thông
                                             qua mã OTP gửi về Email đã đăng
                                             ký.
  ---------------------------------------------------------------------------

## Mô tả chi tiết từng tác nhân

  -------------- ---------------------------------------------------------
  **Tên tác      **Công việc/vai trò**
  nhân**         

  **Khách vãng   Người dùng truy cập vào hệ thống nhưng chưa có tài khoản
  lai (Guest)**  hoặc chưa đăng nhập. Họ có quyền tự do tham quan trang
                 chủ, tìm kiếm sản phẩm, đọc chi tiết mô tả hàng hóa, đọc
                 các bài viết tin tức và xem các chương trình khuyến mãi.
                 Tuy nhiên, họ không được phép thao tác đặt hàng, thêm giỏ
                 hàng hay đánh giá. Muốn thực hiện giao dịch, họ buộc phải
                 đăng ký/đăng nhập.

  **Khách hàng   Là Khách vãng lai đã thực hiện đăng ký và đăng nhập thành
  (Customer)**   công. Họ có toàn quyền thực hiện các luồng mua sắm: thêm
                 sản phẩm vào giỏ, tương tác Chatbot, đặt hàng, thanh toán
                 trực tuyến, tích lũy/sử dụng xu thưởng và gửi yêu cầu
                 hoàn tiền/đánh giá sau khi nhận hàng thành công.

  **Người bán    Cá nhân/doanh nghiệp sở hữu gian hàng trên hệ thống. Đóng
  (Seller)**     vai trò là nhà cung cấp hàng hóa, chịu trách nhiệm đăng
                 tải sản phẩm, thiết kế giao diện gian hàng, xuất bản tin
                 tức của shop và trực tiếp đóng gói, cập nhật trạng thái
                 giao hàng cho khách.

  **Quản trị     Người quản lý cấp cao nhất của hệ thống nền tảng. Đóng
  viên (Admin)** vai trò kiểm duyệt (tài khoản seller, luồng tiền hoàn
                 trả, tranh chấp), duy trì cấu hình giao diện trang chủ,
                 cấu hình hệ thống xu thưởng, và phát hành các chiến dịch
                 Marketing chung toàn sàn (Coupons, Deals, Tin tức hệ
                 thống).
  -------------- ---------------------------------------------------------

Bảng 2‑2: Mô tả các tác nhân

## Sơ đồ Use case

## Đặc tả Use case

### Use case 1

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC01                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Khám phá nền tảng                                     |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người dùng (Khách vãng lai hoặc Khách hàng),   |
| escription** | tôi muốn tham quan trang chủ, tìm kiếm, lọc và xem    |
|              | chi tiết sản phẩm để tìm được món hàng ưng ý.         |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách vãng lai (Guest), Khách hàng (Customer)         |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng truy cập vào URL của hệ thống thông qua    |
|              | trình duyệt web.                                      |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Thiết bị của người dùng có kết nối Internet.          |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Người dùng xem được các thông tin sản phẩm công khai  |
| ndition(s)** | trên nền tảng.                                        |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người dùng truy cập vào nền tảng.                 |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị Trang chủ với các chiến dịch    |
|              | Khuyến mãi (Deals), Banner quảng cáo và Lưới danh mục |
|              | nổi bật.                                              |
|              |                                                       |
|              | 3\. Người dùng nhập từ khóa vào thanh tìm kiếm hoặc   |
|              | sử dụng bộ lọc nâng cao (khoảng giá, màu sắc, danh    |
|              | mục).                                                 |
|              |                                                       |
|              | 4\. Hệ thống truy xuất dữ liệu và trả về danh sách    |
|              | các sản phẩm đáp ứng tiêu chí.                        |
|              |                                                       |
|              | 5\. Người dùng nhấp chọn một sản phẩm cụ thể.         |
|              |                                                       |
|              | 6\. Hệ thống hiển thị màn hình Chi tiết sản phẩm bao  |
|              | gồm: hình ảnh, giá gốc, giá bán, mô tả, đánh giá (chỉ |
|              | đọc) và danh sách các \"Sản phẩm liên quan\" ở cuối   |
|              | trang.                                                |
+--------------+-------------------------------------------------------+
| *            | 3a. Người dùng nhấp vào các liên kết thông tin tĩnh.  |
| *Alternative |                                                       |
| Flow**       | 3a1. Hệ thống điều hướng và hiển thị nội dung các     |
|              | trang: FAQ, Tin tức, Chính sách giao hàng, Chính sách |
|              | hoàn trả.                                             |
|              |                                                       |
|              | Use Case kết thúc.                                    |
+--------------+-------------------------------------------------------+
| **Exception  | 4a. Hệ thống không tìm thấy sản phẩm nào khớp với từ  |
| Flow**       | khóa/bộ lọc.                                          |
|              |                                                       |
|              | 4a1. Hệ thống hiển thị thông báo \"Không tìm thấy sản |
|              | phẩm phù hợp\" và gợi ý xóa bộ lọc.                   |
|              |                                                       |
|              | Use Case quay lại bước 3.                             |
|              |                                                       |
|              | 6b. Khách vãng lai (Guest) cố tình nhấn nút \"Thêm    |
|              | vào giỏ hàng\" hoặc \"Mua ngay\".                     |
|              |                                                       |
|              | 6b1. Hệ thống chặn thao tác và hiển thị yêu cầu đăng  |
|              | nhập/đăng ký.                                         |
|              |                                                       |
|              | Use Case kết thúc.                                    |
+--------------+-------------------------------------------------------+
| **Business   | \- BR01-1 (QĐ_KVL3): Mục \"Sản phẩm liên quan\" bắt   |
| Rules**      | buộc phải truy xuất các sản phẩm có cùng danh mục cấp |
|              | 3 (Level 3 Category) với sản phẩm đang xem.           |
|              |                                                       |
|              | \- BR01-2 (QĐ_KVL5 & 6): Khách vãng lai tuyệt đối     |
|              | không có quyền tạo Giỏ hàng, Wishlist hay Gửi đánh    |
|              | giá.                                                  |
+--------------+-------------------------------------------------------+
| **No         | \- NFR01-1: Tốc độ tải Trang chủ và Trang chi tiết    |
| n-Functional | sản phẩm phải dưới 2 giây dù có lưu lượng truy cập    |
| R            | lớn.                                                  |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 2

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC02                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Đăng ký tài khoản                                     |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách vãng lai, tôi muốn đăng ký tài khoản     |
| escription** | thành viên thông qua mã xác thực OTP để có thể thực   |
|              | hiện mua sắm trên sàn.                                |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách vãng lai (Guest)                                |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng nhấn nút \"Đăng ký\" trên giao diện điều   |
|              | hướng.                                                |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người dùng chưa đăng nhập vào hệ thống.               |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Tài khoản được tạo thành công, người dùng được chuyển |
| ndition(s)** | trạng thái thành Khách hàng (ROLE_CUSTOMER) và được   |
|              | cấp phiên làm việc.                                   |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người dùng chọn \"Đăng ký\" trên màn hình.        |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu yêu cầu cung cấp thông |
|              | tin cơ bản (họ tên, ngày sinh, giới tính) và Email.   |
|              |                                                       |
|              | 3\. Người dùng nhập Email và nhấn \"Gửi mã OTP\".     |
|              |                                                       |
|              | 4\. Hệ thống kiểm tra tính hợp lệ và gửi mã OTP gồm 6 |
|              | chữ số đến Email của người dùng.                      |
|              |                                                       |
|              | 5\. Người dùng nhập mã OTP.                           |
|              |                                                       |
|              | 6\. Người dùng chọn \"Tạo tài khoản\".                |
|              |                                                       |
|              | 7\. Hệ thống xác thực mã OTP. Nếu hợp lệ, hệ thống    |
|              | khởi tạo tài khoản Khách hàng mới.                    |
|              |                                                       |
|              | 8\. Hệ thống thông báo đăng ký thành công, tự động    |
|              | đăng nhập và đưa người dùng về Trang chủ.             |
+--------------+-------------------------------------------------------+
| *            | **3a.** Người dùng chọn lệnh \"Chuyển sang Đăng       |
| *Alternative | nhập\".                                               |
| Flow**       |                                                       |
|              | **3a1.** Hệ thống chuyển đổi biểu mẫu sang màn hình   |
|              | Đăng nhập.                                            |
|              |                                                       |
|              | *Use Case chuyển tiếp sang Use Case Đăng nhập.*       |
+--------------+-------------------------------------------------------+
| **Exception  | **4a.** Email người dùng cung cấp không đúng định     |
| Flow**       | dạng.                                                 |
|              |                                                       |
|              | **4a1.** Hệ thống báo lỗi ngay tại ô nhập liệu và     |
|              | chặn lệnh gửi OTP.                                    |
|              |                                                       |
|              | *Use Case quay lại bước 3.*                           |
|              |                                                       |
|              | **7b.** Người dùng nhập mã OTP sai hoặc mã đã hết hạn |
|              | hiệu lực.                                             |
|              |                                                       |
|              | **7b1.** Hệ thống hiển thị cảnh báo \"Mã OTP không    |
|              | chính xác hoặc đã hết hạn\".                          |
|              |                                                       |
|              | *Use Case quay lại bước 5.*                           |
+--------------+-------------------------------------------------------+
| **Business   | \- BR02-1: Mã OTP chỉ bao gồm 6 chữ số ngẫu nhiên và  |
| Rules**      | có thời gian hiệu lực giới hạn.                       |
+--------------+-------------------------------------------------------+
| **No         | \- NFR02-1: Hệ thống gửi Email chứa mã OTP trong thời |
| n-Functional | gian không quá 5 giây kể từ lúc nhấn nút.             |
| R            |                                                       |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 3

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC03                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Giỏ hàng (Cart)                               |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn thêm sản phẩm, cập nhật   |
| escription** | số lượng hoặc xóa sản phẩm ra khỏi giỏ hàng của mình  |
|              | để chuẩn bị cho việc thanh toán.                      |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng nhấn \"Thêm vào giỏ\" tại một sản phẩm     |
|              | hoặc truy cập trực tiếp vào màn hình Giỏ hàng.        |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập thành công.                   |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Hệ thống ghi nhận đúng số lượng, tổng tiền và áp dụng |
| ndition(s)** | mã giảm giá (nếu có) vào giỏ hàng.                    |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng lựa chọn một sản phẩm và chỉ định thao |
| Flow**       | tác (Thêm mới, Tăng/Giảm số lượng, hoặc Xóa).         |
|              |                                                       |
|              | 2\. Hệ thống kiểm tra quyền hợp lệ và số lượng tồn    |
|              | kho của sản phẩm tương ứng.                           |
|              |                                                       |
|              | 3\. Nếu hợp lệ, hệ thống thực thi việc cập nhật giỏ   |
|              | hàng theo yêu cầu.                                    |
|              |                                                       |
|              | 4\. Hệ thống tự động tính toán lại Tổng tiền gốc      |
|              | (Total MRP), Tổng tiền thanh toán (Total Selling      |
|              | Price) và Tổng số lượng món hàng.                     |
|              |                                                       |
|              | 5\. Hệ thống hiển thị thông báo cập nhật thành công   |
|              | và hiển thị giao diện Giỏ hàng mới nhất.              |
+--------------+-------------------------------------------------------+
| *            | *Không có rẽ nhánh.*                                  |
| *Alternative |                                                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | **2a.** Khách hàng muốn tăng số lượng nhưng sản phẩm  |
| Flow**       | đó đã đạt mức tối đa tồn kho (Out of stock).          |
|              |                                                       |
|              | **2a1.** Hệ thống báo lỗi không thể thêm số lượng và  |
|              | chặn thao tác.                                        |
|              |                                                       |
|              | *Use Case dừng lại.*                                  |
|              |                                                       |
|              | **1a1.1** (Ngoại lệ của 1a): Mã giảm giá đã hết hạn,  |
|              | nhập sai, hoặc chưa đạt giá trị tối thiểu.            |
|              |                                                       |
|              | **1a1.2.** Hệ thống hiển thị thông báo \"Mã giảm giá  |
|              | không hợp lệ hoặc đã được sử dụng\".                  |
|              |                                                       |
|              | *Use Case quay lại màn hình Giỏ hàng.*                |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR03-1:** Mỗi Khách hàng chỉ sở hữu duy nhất 1   |
| Rules**      | Giỏ hàng (Mối quan hệ 1-1).                           |
|              |                                                       |
|              | \- **BR03-2:** Việc áp dụng Mã giảm giá ở Giỏ hàng    |
|              | mang tính chất **Ước lượng (Estimated)**. Mã này sẽ   |
|              | được hệ thống mang sang bước Thanh toán (UC05) để     |
|              | quét lại một lần nữa (kiểm tra chéo với địa chỉ giao  |
|              | nhận và hình thức thanh toán) trước khi chốt giá cuối |
|              | cùng.                                                 |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR03-1:** Tính toán thay đổi tổng tiền phải     |
| n-Functional | diễn ra tức thì (Real-time logic) trên giao diện ngay |
| R            | khi khách hàng nhấn nút +/-.                          |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 4

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC04                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Danh sách yêu thích (Wishlist)                |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn lưu lại các sản phẩm mà   |
| escription** | mình quan tâm vào một danh sách riêng để dễ dàng theo |
|              | dõi và đặt mua vào lần sau.                           |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng nhấn vào biểu tượng \"Trái tim\" trên thẻ  |
|              | sản phẩm hoặc truy cập menu Wishlist.                 |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập thành công.                   |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Sản phẩm được thêm vào hoặc loại bỏ khỏi Wishlist.    |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng thao tác nhấn biểu tượng \"Trái tim\"  |
| Flow**       | trên một sản phẩm.                                    |
|              |                                                       |
|              | 2\. Hệ thống kiểm tra sản phẩm này đã tồn tại trong   |
|              | Danh sách yêu thích của khách hàng hay chưa.          |
|              |                                                       |
|              | 3\. Hệ thống ghi nhận sản phẩm vào Cơ sở dữ liệu      |
|              | Wishlist của khách hàng.                              |
|              |                                                       |
|              | 4\. Hệ thống đổi màu biểu tượng \"Trái tim\"          |
|              | (highlight) và hiển thị thông báo đã thêm thành công. |
+--------------+-------------------------------------------------------+
| *            | 2a. Hệ thống phát hiện sản phẩm ĐÃ TỒN TẠI trong Danh |
| *Alternative | sách yêu thích.                                       |
| Flow**       |                                                       |
|              | 2a1. Hệ thống hiểu đây là lệnh Xóa (Remove).          |
|              |                                                       |
|              | 2a2. Hệ thống gỡ sản phẩm ra khỏi Wishlist, bỏ        |
|              | highlight biểu tượng \"Trái tim\" và thông báo đã xóa |
|              | thành công.                                           |
|              |                                                       |
|              | *Use Case kết thúc.*                                  |
|              |                                                       |
|              | 1b. Khách hàng truy cập trang quản lý Danh sách yêu   |
|              | thích.                                                |
|              |                                                       |
|              | 1b1. Hệ thống truy xuất và hiển thị lưới toàn bộ sản  |
|              | phẩm khách hàng đã lưu.                               |
|              |                                                       |
|              | 1b2. Khách hàng nhấp vào icon \"Dấu X\" (Close) trên  |
|              | thẻ sản phẩm.                                         |
|              |                                                       |
|              | *Use Case tiếp tục thực hiện luồng 2a2.*              |
+--------------+-------------------------------------------------------+
| **Exception  | *Xử lý lỗi rớt mạng (Timeout).*                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR04-1:** Mỗi người dùng chỉ có đúng 1 Danh sách |
| Rules**      | yêu thích (1-1), một sản phẩm chỉ được xuất hiện 1    |
|              | lần trong danh sách đó.                               |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR04-1:** Trạng thái thêm/xóa khỏi Wishlist     |
| n-Functional | phải được lưu động bộ vào cơ sở dữ liệu và hiển thị   |
| R            | phản hồi giao diện không cần tải lại trang.           |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 5

  --------------------------------------------------------------------------------
  Trường                  Nội dung
  ----------------------- --------------------------------------------------------
  **Use Case ID**         UC05

  **Use Case Name**       Đặt hàng và Thanh toán

  **Description**         Là một Khách hàng, tôi muốn chọn địa chỉ, phương thức
                          thanh toán (trực tuyến hoặc COD) và hệ thống sẽ tự động
                          đề xuất/áp dụng mã giảm giá tốt nhất cho tôi trước khi
                          tôi trả tiền.

  **Actor(s)**            Khách hàng (Customer)

  **Priority**            Must Have

  **Trigger**             Khách hàng nhấn nút \"Checkout\" (Tiến hành đặt hàng)
                          tại màn hình Giỏ hàng.

  **Pre-Condition(s)**    Khách hàng đã đăng nhập và Giỏ hàng đang chứa ít nhất 1
                          sản phẩm.

  **Post-Condition(s)**   Đơn hàng được tạo thành công, hàng trong kho bị trừ,
                          giao dịch thanh toán được ghi nhận (hoặc ghi nhận chờ
                          thu hộ đối với COD).

  **Basic Flow**          1\. Khách hàng bắt đầu tiến trình Checkout.\
                          2. Hệ thống hiển thị màn hình Thanh toán gồm: Sổ địa
                          chỉ, Phương thức thanh toán (VNPay/SePay/Momo/COD), Danh
                          sách sản phẩm, và Khung tóm tắt đơn hàng.\
                          3. Khách hàng lựa chọn 1 Địa chỉ giao nhận hiện có và
                          chọn 1 Phương thức thanh toán.\
                          4. \[Voucher Engine\]: Dựa trên Tổng tiền hàng, Địa chỉ
                          giao nhận (tính Phí ship) và Phương thức thanh toán vừa
                          chọn, hệ thống tự động quét và áp dụng Mã giảm giá
                          (Coupon) mang lại mức chiết khấu cao nhất cho khách
                          hàng.\
                          5. (Tùy chọn) Khách hàng nhập yêu cầu áp dụng Xu tích
                          lũy vào đơn hàng.\
                          6. Hệ thống tính toán tổng thanh toán cuối cùng (Final
                          Payment).\
                          7. Khách hàng nhấn lệnh \"Xác nhận và Thanh toán\" (Pay
                          Now).\
                          8. Hệ thống tự động Tách giỏ hàng (Split Order): gom
                          nhóm sản phẩm theo Seller thành các Đơn hàng phụ, nhưng
                          gộp chung giá trị vào một lệnh thanh toán.\
                          9. \[Thanh toán trực tuyến\]: Khách hàng hoàn tất nhập
                          thông tin tại cổng thanh toán bảo mật của bên thứ 3.\
                          10. Webhook trả kết quả về hệ thống. Đơn hàng chuyển
                          trạng thái \"Đã đặt\" (PLACED), hiển thị màn hình \"Giao
                          dịch thành công\".

  **Alternative Flow**    **3a.** Khách hàng chọn \"Thêm địa chỉ mới\".\
                          3a1. Khách hàng điền biểu mẫu thông tin và lưu lại. Hệ
                          thống lưu vào Sổ địa chỉ.\
                          Use Case quay lại bước 3 và tự động chạy lại bước 4.\
                          **4a.** Khách hàng muốn đổi mã giảm giá khác:\
                          4a1. Khách hàng nhấn vào mục \"Chọn mã giảm giá\".\
                          4a2. Khách hàng chọn 1 mã khác từ Kho Voucher hoặc nhập
                          mã thủ công và nhấn Áp dụng.\
                          → Use Case đi tiếp đến bước 5.\
                          **9b. \[Thanh toán COD\]:** Khách hàng đã chọn phương
                          thức \"Thanh toán khi nhận hàng\" (COD) ở bước 3.\
                          9b1. Hệ thống KHÔNG chuyển hướng sang cổng thanh toán mà
                          tạo đơn hàng trực tiếp với trạng thái PLACED và
                          PaymentOrder ở trạng thái COD_PENDING (Chờ thu hộ).\
                          9b2. Hệ thống trừ hàng tồn kho, xóa giỏ hàng, gửi thông
                          báo cho Seller.\
                          9b3. Hiển thị màn hình \"Đặt hàng thành công\".\
                          → Khi Shipper giao hàng và thu tiền mặt, trạng thái
                          PaymentOrder sẽ được ĐVVC cập nhật qua Webhook sang
                          COD_COLLECTED. Use Case kết thúc.

  **Exception Flow**      **4a2.1** (Ngoại lệ của 4a): Khách hàng nhập thủ công
                          một mã giảm giá đã hết hạn, hoặc không thỏa mãn điều
                          kiện Phương thức thanh toán/Giá trị tối thiểu.\
                          4a2.2: Hệ thống báo lỗi \"Mã không hợp lệ hoặc chưa đủ
                          điều kiện\" và từ chối áp dụng.\
                          Use Case quay lại bước 4.\
                          **9a.** Thanh toán trực tuyến thất bại hoặc khách hàng
                          hủy giao dịch tại Cổng thanh toán.\
                          9a1. Hệ thống ghi nhận thanh toán thất bại, không tạo
                          đơn hàng thành công.\
                          Use Case dừng lại.

  **Business Rules**      - **BR05-1** (Thuật toán Auto-apply): Khi có sự thay đổi
                          về Địa chỉ hoặc Phương thức thanh toán, Voucher Engine
                          bắt buộc phải quét lại toàn bộ dữ liệu để cập nhật lại
                          mã giảm giá và phí ship theo thời gian thực.\
                          - **BR05-2** (QĐ_KH4): Bắt buộc tách Order riêng biệt
                          theo mã Seller_ID.\
                          - **BR05-3** (CT_KH1): Final Payment = (Tổng tiền hàng)
                          − (Coupon) − (Quy đổi Xu) + (Phí vận chuyển).\
                          - **BR05-4** (COD): Đơn hàng COD không được phép sử dụng
                          Xu tích lũy (chỉ cho phép Coupon). Một số Coupon có thể
                          giới hạn chỉ áp dụng cho thanh toán trực tuyến. Phí thu
                          hộ COD (nếu có) được ĐVVC tính thêm vào Phí vận chuyển.

  **Non-Functional        - **NFR05-1**: Thuật toán quét và áp dụng Voucher phải
  Requirement**           được xử lý ở tốc độ cao (dưới 1s) để không gây giật lag
                          giao diện (giữ UX mượt mà).\
                          - **NFR05-2**: Thông tin thẻ không lưu trên DB hệ thống,
                          truyền mã hóa 100% qua cổng VnPay/SePay/Momo.
  --------------------------------------------------------------------------------

### Use case 6

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC06                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Theo dõi & Quản lý đơn hàng                           |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn xem lại lịch sử các đơn   |
| escription** | đã đặt, tra cứu tiến trình giao hàng theo thời gian   |
|              | thực và hủy đơn nếu thay đổi ý định mua sắm.          |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng truy cập vào mục \"Đơn hàng của tôi\" (My  |
|              | Orders) trên giao diện.                               |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập thành công vào hệ thống.      |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Khách hàng nắm bắt được trạng thái đơn hàng. Nếu thực |
| ndition(s)** | hiện thao tác hủy, hệ thống ghi nhận trạng thái Hủy   |
|              | (Canceled) và hoàn lại kho.                           |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng chọn mục \"Đơn hàng của tôi\".         |
| Flow**       |                                                       |
|              | 2\. Hệ thống truy xuất và hiển thị danh sách toàn bộ  |
|              | đơn hàng (lịch sử mua sắm).                           |
|              |                                                       |
|              | 3\. Khách hàng nhấp vào một đơn hàng cụ thể để xem    |
|              | chi tiết.                                             |
|              |                                                       |
|              | 4\. Hệ thống hiển thị màn hình Chi tiết Đơn hàng, bao |
|              | gồm thanh tiến trình (Order Stepper) trực quan với    |
|              | các mốc: Đã đặt (Placed), Đã xác nhận (Confirmed),    |
|              | Đang giao (Shipped), và Đã giao (Delivered).          |
|              |                                                       |
|              | 5\. Khách hàng theo dõi vị trí và tiến trình. Thao    |
|              | tác xem hoàn tất.                                     |
+--------------+-------------------------------------------------------+
| *            | **3a. Khách hàng muốn hủy đơn hàng:**                 |
| *Alternative |                                                       |
| Flow**       | 3a1. Tại màn hình chi tiết, khách hàng chọn lệnh      |
|              | \"Hủy đơn hàng\" (Cancel Order).                      |
|              |                                                       |
|              | 3a2. Hệ thống yêu cầu xác nhận. Khách hàng đồng ý.    |
|              |                                                       |
|              | 3a3. Hệ thống đổi trạng thái đơn sang \"Đã hủy\"      |
|              | (Canceled), tự động cộng lại số lượng sản phẩm vào    |
|              | kho tồn của Seller và thông báo hủy thành công.       |
+--------------+-------------------------------------------------------+
| **Exception  | **3a1.1 (Ngoại lệ của luồng Hủy đơn):** Đơn hàng đã   |
| Flow**       | chuyển sang trạng thái \"Đang giao\" (Shipped) hoặc   |
|              | \"Đã giao\" (Delivered).                              |
|              |                                                       |
|              | 3a1.2. Hệ thống tự động vô hiệu hóa (disable) hoặc ẩn |
|              | nút \"Hủy đơn hàng\", khách hàng không thể thực hiện  |
|              | thao tác này.                                         |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR06-1 (QĐ_KH6):** Trạng thái trên thanh tiến    |
| Rules**      | trình (Order Stepper) được cập nhật tự động và đồng   |
|              | bộ từ Đơn vị vận chuyển. Khách hàng không cần rời     |
|              | khỏi sàn để tra cứu.                                  |
|              |                                                       |
|              | \- **BR06-2:** Chỉ cho phép Khách hàng hủy đơn khi    |
|              | đơn hàng đang ở trạng thái \"Pending\" hoặc           |
|              | \"Placed\".                                           |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR06-1:** Lịch sử đơn hàng của khách hàng phải  |
| n-Functional | được lưu trữ trên hệ thống tối thiểu 12 tháng để tra  |
| R            | cứu đối soát.                                         |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 7

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC07                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Tương tác Chatbot AI                                  |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người dùng, tôi muốn chat bằng ngôn ngữ tự     |
| escription** | nhiên với trợ lý ảo AI để tìm hiểu tổng quan về hệ    |
|              | thống, xu hướng sản phẩm, hoặc tra cứu tình trạng đơn |
|              | hàng/giỏ hàng cá nhân.                                |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách vãng lai (Guest), Khách hàng (Customer)         |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng nhấp vào biểu tượng Chatbot nổi ở góc màn  |
|              | hình.                                                 |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Hệ thống AI Server đang hoạt động bình thường.        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Người dùng nhận được câu trả lời chính xác dựa trên   |
| ndition(s)** | kho dữ liệu và ngữ cảnh định danh của hệ thống.       |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người dùng mở khung Chatbot AI.                   |
| Flow**       |                                                       |
|              | 2\. Người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên    |
|              | (Ví dụ: *\"Nền tảng này bán gì?\", \"Sản phẩm nào     |
|              | đang hot nhất?\",* hoặc *\"Đơn hàng của tôi đâu?\"*). |
|              |                                                       |
|              | 3\. Khách hàng nhấn Gửi.                              |
|              |                                                       |
|              | 4\. Hệ thống tiếp nhận, AI phân tích ý định (intent)  |
|              | của câu hỏi.                                          |
|              |                                                       |
|              | 5\. Hệ thống nhận diện trạng thái đăng nhập và truy   |
|              | xuất dữ liệu phù hợp để trả lời.                      |
|              |                                                       |
|              | 6\. Chatbot phản hồi lại tin nhắn cho người dùng kèm  |
|              | theo thông tin chi tiết.                              |
+--------------+-------------------------------------------------------+
| *            | 2a. Người dùng đang ở trong trang Chi tiết Sản phẩm   |
| *Alternative | và mở Chatbot hỏi về sản phẩm đó (Ví dụ: *\"Sản phẩm  |
| Flow**       | này tôi được giảm giá bao nhiêu?\"*).                 |
|              |                                                       |
|              | 2a1. Hệ thống AI tự động bắt ngữ cảnh của ID sản phẩm |
|              | đang xem và trả về đúng thông số % giảm giá, màu sắc  |
|              | của sản phẩm đó.                                      |
+--------------+-------------------------------------------------------+
| **Exception  | 5a. Phân quyền dữ liệu cá nhân:                       |
| Flow**       |                                                       |
|              | 5a1. Khách vãng lai (Guest) đặt câu hỏi liên quan đến |
|              | dữ liệu cá nhân (Giỏ hàng, Đơn hàng, Ví xu).          |
|              |                                                       |
|              | 5a2. Chatbot từ chối trả lời, yêu cầu Người dùng phải |
|              | Đăng nhập và hiển thị kèm nút \"Đi đến trang Đăng     |
|              | nhập\".                                               |
|              |                                                       |
|              | 5b. AI không hiểu câu hỏi:                            |
|              |                                                       |
|              | 5b1. Câu hỏi nằm ngoài phạm vi dữ liệu hệ thống (Ví   |
|              | dụ: Hỏi về thời tiết).                                |
|              |                                                       |
|              | 5b2. Hệ thống phản hồi lại thông báo từ chối khéo léo |
|              | và hướng dẫn khách hàng hỏi lại các vấn đề liên quan  |
|              | đến mua sắm.                                          |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR07-1:** Nếu là Khách vãng lai, Chatbot chỉ     |
| Rules**      | được phép truy xuất kho dữ liệu Public (Tổng quan hệ  |
|              | thống, FAQ, Sản phẩm tìm kiếm nhiều nhất, Khuyến      |
|              | mãi).                                                 |
|              |                                                       |
|              | \- **BR07-2:** Nếu là Khách hàng (đã đăng nhập),      |
|              | Chatbot được cấp thêm quyền truy xuất dữ liệu Private |
|              | (Đơn hàng, Giỏ hàng, Lịch sử) thuộc sở hữu của chính  |
|              | User đó.                                              |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR07-1:** Thời gian phản hồi của Chatbot AI (AI |
| n-Functional | processing time) không được vượt quá 3 giây để đảm    |
| R            | bảo tính thời gian thực.                              |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 8

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC08                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Đánh giá sản phẩm                                     |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn chấm điểm và viết nhận    |
| escription** | xét (review) cho món hàng mình đã mua để phản hồi     |
|              | chất lượng cho Người bán và những người mua sau.      |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng nhấn nút \"Viết đánh giá\" (Write review)  |
|              | tại màn hình Lịch sử đơn hàng.                        |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập và Đơn hàng chứa sản phẩm đó  |
| ndition(s)** | phải ở trạng thái \"Đã giao\" (DELIVERED).            |
+--------------+-------------------------------------------------------+
| **Post-Co    | Bài đánh giá được lưu lại và hiển thị công khai trên  |
| ndition(s)** | trang chi tiết sản phẩm.                              |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng chọn lệnh Viết đánh giá cho một sản    |
| Flow**       | phẩm đã nhận.                                         |
|              |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu (Form) đánh giá.       |
|              |                                                       |
|              | 3\. Khách hàng chọn số sao mong muốn (Rating từ 1 đến |
|              | 5 sao).                                               |
|              |                                                       |
|              | 4\. Khách hàng nhập nội dung bình luận (Review text)  |
|              | và tải lên các hình ảnh thực tế của sản phẩm.         |
|              |                                                       |
|              | 5\. Khách hàng chọn lệnh \"Gửi đánh giá\".            |
|              |                                                       |
|              | 6\. Hệ thống lưu hình ảnh lên Cloudinary, ghi nhận    |
|              | đánh giá vào cơ sở dữ liệu.                           |
|              |                                                       |
|              | 7\. Hệ thống thông báo thành công và cập nhật lại     |
|              | điểm đánh giá trung bình của sản phẩm đó.             |
+--------------+-------------------------------------------------------+
| *            | 1a. Khách hàng muốn xóa bài đánh giá:                 |
| *Alternative |                                                       |
| Flow**       | 1a1. Khách hàng truy cập lại bài đánh giá mình đã     |
|              | viết và chọn lệnh \"Xóa\".                            |
|              |                                                       |
|              | 1a2. Hệ thống yêu cầu xác nhận. Khách hàng đồng ý.    |
|              |                                                       |
|              | 1a3. Hệ thống xóa bài đánh giá và tính lại điểm sao   |
|              | trung bình.                                           |
+--------------+-------------------------------------------------------+
| **Exception  | **4a.** Khách hàng không chọn số sao (để trống        |
| Flow**       | Rating).                                              |
|              |                                                       |
|              | 4a1. Hệ thống báo lỗi \"Vui lòng chọn số sao đánh     |
|              | giá\" và chặn thao tác gửi.                           |
|              |                                                       |
|              | **6a.** Quá trình tải ảnh lên máy chủ thất bại do sai |
|              | định dạng hoặc quá dung lượng.                        |
|              |                                                       |
|              | 6a1. Hệ thống báo lỗi và yêu cầu tải lại hình ảnh.    |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR08-1 (QĐ_KH7):** Hệ thống chỉ kích hoạt nút    |
| Rules**      | \"Viết đánh giá\" khi trạng thái đơn hàng của sản     |
|              | phẩm đó chính xác là DELIVERED. Khách chưa mua hoặc   |
|              | chưa nhận hàng tuyệt đối không được đánh giá.         |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR08-1:** Hệ thống xử lý cập nhật lại điểm đánh |
| n-Functional | giá trung bình (Average Rating) của sản phẩm ngay lập |
| R            | tức sau khi Submit.                                   |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 9

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC09                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Yêu cầu Trả hàng & Hoàn tiền                          |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn yêu cầu hệ thống cho trả  |
| escription** | lại hàng lỗi và hoàn lại tiền, đồng thời có thể khiếu |
|              | nại lên Ban Quản trị nếu bị Người bán làm khó dễ.     |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng nhấn nút \"Yêu cầu Trả hàng/Hoàn tiền\"    |
|              | tại đơn hàng.                                         |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Đơn hàng phải ở trạng thái \"Đã giao\" (DELIVERED) và |
| ndition(s)** | nằm trong thời gian quy định (Ví dụ: 7 ngày kể từ khi |
|              | nhận).                                                |
+--------------+-------------------------------------------------------+
| **Post-Co    | Đơn hàng bị đóng băng dòng tiền đối soát và chuyển    |
| ndition(s)** | sang trạng thái chờ xử lý (RETURN_REQUESTED hoặc      |
|              | DISPUTED).                                            |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng truy cập lịch sử mua hàng, chọn một    |
| Flow**       | đơn hàng đủ điều kiện và chọn lệnh \"Yêu cầu trả      |
|              | hàng\".                                               |
|              |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu yêu cầu cung cấp minh  |
|              | chứng.                                                |
|              |                                                       |
|              | 3\. Khách hàng điền lý do chi tiết (VD: Hàng lỗi,     |
|              | giao sai màu) và tải lên hình ảnh/video khui hàng.    |
|              |                                                       |
|              | 4\. Khách hàng chọn gửi yêu cầu.                      |
|              |                                                       |
|              | 5\. Hệ thống xác nhận, chuyển trạng thái đơn hàng     |
|              | sang RETURN_REQUESTED và gửi thông báo cho Người bán  |
|              | xử lý.                                                |
|              |                                                       |
|              | 6\. Tùy thuộc vào quyết định của Người bán, nếu chấp  |
|              | nhận, quy trình trả hàng diễn ra và Khách hàng được   |
|              | hoàn tiền về tài khoản.                               |
+--------------+-------------------------------------------------------+
| *            | 6a. Khiếu nại (Dispute): Nếu Người bán ấn từ chối yêu |
| *Alternative | cầu trả hàng ở Bước 6.                                |
| Flow**       |                                                       |
|              | 6a1. Hệ thống báo kết quả từ chối về cho Khách hàng.  |
|              |                                                       |
|              | 6a2. Khách hàng chọn lệnh \"Khiếu nại lên Admin\"     |
|              | (Escalate to Admin).                                  |
|              |                                                       |
|              | 6a3. Hệ thống ghi nhận khiếu nại và đổi trạng thái    |
|              | đơn hàng sang DISPUTED (Đang tranh chấp).             |
|              |                                                       |
|              | 6a4. Quản trị viên (Admin) sẽ vào làm trọng tài để    |
|              | phán quyết cuối cùng dựa trên các bằng chứng từ Khách |
|              | và Seller.                                            |
+--------------+-------------------------------------------------------+
| **Exception  | 1a. Đơn hàng đã quá thời hạn 7 ngày kể từ ngày nhận   |
| Flow**       | hàng.                                                 |
|              |                                                       |
|              | 1a1. Hệ thống ẩn nút \"Yêu cầu trả hàng\", khách hàng |
|              | không thể thao tác.                                   |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR09-1 (QĐ_KH9):** Policy thời hạn hoàn trả được |
| Rules**      | cấu hình là 7 ngày kể từ khi trạng thái là DELIVERED. |
|              | Qua ngày thứ 8, hệ thống tự động khóa tính năng này.  |
|              |                                                       |
|              | \- **BR09-2 (QĐ_KH10):** Khi đơn hàng bị đẩy lên      |
|              | trạng thái DISPUTED, quyết định của Admin trên hệ     |
|              | thống là kết quả bắt buộc cuối cùng.                  |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR09-1:** Việc lưu trữ hình ảnh/video bằng      |
| n-Functional | chứng (Evidences) phải được tối ưu nén trên           |
| R            | Cloudinary để giảm tải database máy chủ.              |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 10

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC10                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Cập nhật thông tin cá nhân                            |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn cập nhật thông tin cá     |
| escription** | nhân của mình, để đảm bảo hệ thống lưu trữ đúng dữ    |
|              | liệu liên lạc phục vụ cho quá trình xác thực và chăm  |
|              | sóc khách hàng.                                       |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng truy cập vào Menu \"Tài khoản của tôi\" và |
|              | chọn tab \"Hồ sơ cá nhân\" (Profile).                 |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập vào hệ thống.                 |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Thông tin cá nhân mới được lưu trữ thành công vào cơ  |
| ndition(s)** | sở dữ liệu.                                           |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng truy cập tab \"Hồ sơ cá nhân\".        |
| Flow**       |                                                       |
|              | 2\. Hệ thống truy xuất dữ liệu hiện tại và hiển thị   |
|              | trên biểu mẫu (Form).                                 |
|              |                                                       |
|              | 3\. Khách hàng thực hiện chỉnh sửa các trường thông   |
|              | tin mong muốn (Họ tên, Số điện thoại).                |
|              |                                                       |
|              | 4\. Khách hàng nhấn nút \"Lưu thay đổi\"              |
|              |                                                       |
|              | 5\. Hệ thống kiểm tra tính hợp lệ của dữ liệu đầu     |
|              | vào.                                                  |
|              |                                                       |
|              | 6\. Hệ thống ghi nhận thông tin mới vào cơ sở dữ      |
|              | liệu.                                                 |
|              |                                                       |
|              | 7\. Hệ thống hiển thị thông báo \"Cập nhật thành      |
|              | công\" và làm mới lại dữ liệu hiển thị.               |
+--------------+-------------------------------------------------------+
| *            | *Không có nhánh rẽ phức tạp, người dùng thao tác trực |
| *Alternative | tiếp trên biểu mẫu.*                                  |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | 5a. Khách hàng bỏ trống trường thông tin bắt buộc (Họ |
| Flow**       | tên) hoặc nhập sai định dạng Số điện thoại.           |
|              |                                                       |
|              | 5a1. Hệ thống hiển thị thông báo lỗi bôi đỏ tại       |
|              | trường tương ứng và chặn lệnh lưu.                    |
|              |                                                       |
|              | *Use Case quay lại bước 3.*                           |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR10-1:** Khách hàng không được phép thay đổi    |
| Rules**      | Email đăng nhập. Trường Email được đặt ở chế độ       |
|              | Read-only (Chỉ đọc) trên giao diện.                   |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR10-1:** Thông tin thay đổi phải được cập nhật |
| n-Functional | ngay lập tức lên Header (Khu vực hiển thị Tên Avatar) |
| R            | nhờ cơ chế quản lý trạng thái (Redux Toolkit) mà      |
| equirement** | không cần tải lại trang.                              |
+--------------+-------------------------------------------------------+

### Use case 11

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC11                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Địa chỉ                                       |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn quản lý danh sách đa địa  |
| escription** | chỉ nhận hàng, để có thể chọn nhanh địa chỉ phù hợp   |
|              | tại bước Thanh toán mà không phải nhập tay lại từ     |
|              | đầu.                                                  |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng truy cập vào tab \"Sổ địa chỉ\" (Saved     |
|              | Addresses) trong phần Quản lý tài khoản.              |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập.                              |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Địa chỉ được Thêm mới, Cập nhật hoặc Xóa thành công   |
| ndition(s)** | khỏi Sổ địa chỉ của khách hàng.                       |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng truy cập tab \"Sổ địa chỉ\".           |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị danh sách các Thẻ địa chỉ       |
|              | (Address Cards) hiện có.\<br\>3. Khách hàng chọn một  |
|              | thao tác:                                             |
|              |                                                       |
|              |    - **Thêm mới:** Nhấn \"Thêm địa chỉ mới\", hệ      |
|              | thống hiển thị form trống.                            |
|              |                                                       |
|              |    - **Cập nhật:** Nhấn \"Chỉnh sửa\" tại một thẻ địa |
|              | chỉ, hệ thống hiển thị form chứa sẵn dữ liệu cũ.      |
|              |                                                       |
|              |    - **Xóa:** Nhấn icon \"Xóa\" tại một thẻ địa chỉ.  |
|              |                                                       |
|              | 4\. Khách hàng điền/sửa thông tin (Tên, SĐT,          |
|              | Tỉnh/Thành, Quận/Huyện, Chi tiết) và xác nhận lệnh.   |
|              |                                                       |
|              | 5\. Hệ thống kiểm tra dữ liệu, ghi nhận thay đổi vào  |
|              | cơ sở dữ liệu.                                        |
|              |                                                       |
|              | 6\. Hệ thống hiển thị thông báo thành công và tự động |
|              | cập nhật lại danh sách Sổ địa chỉ.                    |
+--------------+-------------------------------------------------------+
| *            | **1a.** Hành động \"Thêm địa chỉ mới\" cũng có thể    |
| *Alternative | được kích hoạt trực tiếp từ màn hình Đặt hàng và      |
| Flow**       | Thanh toán (Checkout) thay vì phải vào Quản lý tài    |
|              | khoản.                                                |
+--------------+-------------------------------------------------------+
| **Exception  | **5a.** Khách hàng thực hiện \"Thêm/Sửa\" nhưng bỏ    |
| Flow**       | trống các trường bắt buộc.                            |
|              |                                                       |
|              | **5a1.** Hệ thống báo lỗi bôi đỏ tại các trường chưa  |
|              | nhập và chặn lệnh lưu.                                |
|              |                                                       |
|              | **3a.** Khách hàng Xóa địa chỉ duy nhất đang tồn tại  |
|              | hoặc địa chỉ đang được sử dụng cho một đơn hàng chưa  |
|              | giao xong.                                            |
|              |                                                       |
|              | **3a1.** Hệ thống hiển thị cảnh báo từ chối xóa để    |
|              | đảm bảo an toàn tiến trình giao hàng.                 |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR11-1 (QĐ_KH14):** Sổ địa chỉ cho phép lưu      |
| Rules**      | nhiều bản ghi. Danh sách này sẽ được gọi ra dưới dạng |
|              | các Address Card tại bước Thanh toán (UC05).          |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR11-1:** Các Form nhập liệu                    |
| n-Functional | Tỉnh/Thành/Phường/Xã nên được thiết kế dưới dạng      |
| R            | Dropdown Select gọi từ API hành chính để đồng bộ      |
| equirement** | chuẩn dữ liệu.                                        |
+--------------+-------------------------------------------------------+

### Use case 12

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC12                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Theo dõi ví xu                                        |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng, tôi muốn theo dõi biến động số dư  |
| escription** | Ví Xu của mình, để biết chính xác số lượng xu tích    |
|              | lũy có thể dùng để giảm giá cho các đơn đặt hàng tiếp |
|              | theo.                                                 |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng truy cập vào tab \"Ví Xu\" (Coins) trong   |
|              | phần Quản lý tài khoản.                               |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Khách hàng đã đăng nhập.                              |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Khách hàng nắm bắt được chính xác số dư hiện tại và   |
| ndition(s)** | lịch sử nhận/trừ xu.                                  |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng truy cập tab \"Ví Xu\".                |
| Flow**       |                                                       |
|              | 2\. Hệ thống gửi yêu cầu truy xuất dữ liệu từ máy     |
|              | chủ.                                                  |
|              |                                                       |
|              | 3\. Hệ thống hiển thị Thẻ tổng quan chứa: Tổng số dư  |
|              | Xu hiện tại (Available Coins).                        |
|              |                                                       |
|              | 4\. Hệ thống hiển thị Danh sách lịch sử biến động     |
|              | (Transaction History) bên dưới, bao gồm các thông     |
|              | tin:                                                  |
|              |                                                       |
|              |    - Trạng thái: Nhận (Earned) hoặc Trừ (Spent).      |
|              |                                                       |
|              |    - Số lượng Xu biến động (+/-).                     |
|              |                                                       |
|              |    - Thời gian thực hiện và Mã đơn hàng tham chiếu.   |
|              |                                                       |
|              | 5\. Khách hàng tra cứu thông tin (Thao tác chỉ đọc).  |
+--------------+-------------------------------------------------------+
| *            | *Không có.*                                           |
| *Alternative |                                                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | *Xử lý lỗi rớt mạng hoặc quá thời gian tải dữ liệu    |
| Flow**       | (Timeout).*                                           |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR12-1 (QĐ_KH11):** Giao diện Ví Xu ở chế độ     |
| Rules**      | Read-only (Chỉ Đọc) đối với Khách hàng. Việc cộng/trừ |
|              | xu được hệ thống tự động kích hoạt dựa trên trạng     |
|              | thái Đơn hàng.                                        |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR12-1:** Lịch sử biến động ví xu cần được sắp  |
| n-Functional | xếp theo thứ tự thời gian mới nhất lên đầu (Sort DESC |
| R            | by Time).                                             |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 13

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC13                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Chat trực tuyến (Real-time)                           |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Khách hàng hoặc Người bán, tôi muốn trò chuyện |
| escription** | trực tiếp với đối phương theo thời gian thực để trao  |
|              | đổi chi tiết, tư vấn và giải đáp thắc mắc về sản      |
|              | phẩm/đơn hàng.                                        |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Khách hàng (Customer), Người bán (Seller)             |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Khách hàng nhấn vào nút \"Chat với Người bán\" tại    |
|              | trang sản phẩm, hoặc Người bán nhấn nút \"Phản hồi    |
|              | Khách hàng\" từ Bảng điều khiển (Dashboard).          |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Cả hai tác nhân đều đã đăng nhập thành công vào hệ    |
| ndition(s)** | thống.                                                |
+--------------+-------------------------------------------------------+
| **Post-Co    | Nội dung đoạn chat được hệ thống lưu trữ làm lịch sử  |
| ndition(s)** | và hiển thị đồng bộ trên thiết bị của cả hai bên.     |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Khách hàng nhấn vào biểu tượng \"Chat\" trên      |
| Flow**       | trang chi tiết sản phẩm hoặc đơn hàng.                |
|              |                                                       |
|              | 2\. Hệ thống thiết lập kênh kết nối WebSockets và     |
|              | hiển thị cửa sổ hộp thoại trò chuyện.                 |
|              |                                                       |
|              | 3\. Khách hàng nhập nội dung tin nhắn và nhấn         |
|              | \"Gửi\".                                              |
|              |                                                       |
|              | 4\. Hệ thống lưu tin nhắn vào cơ sở dữ liệu và ngay   |
|              | lập tức đẩy (push) tin nhắn đó đến Bảng điều khiển    |
|              | của Người bán tương ứng.                              |
|              |                                                       |
|              | 5\. Người bán nhận được thông báo tin nhắn mới, mở    |
|              | hộp thoại trò chuyện và gõ câu trả lời.               |
|              |                                                       |
|              | 6\. Hệ thống đẩy tin nhắn phản hồi, hiển thị tức thì  |
|              | (Real-time) lên màn hình của Khách hàng mà không cần  |
|              | phải tải lại trang web.                               |
+--------------+-------------------------------------------------------+
| *            | 1a. Người bán chủ động khởi tạo cuộc trò chuyện với   |
| *Alternative | Khách hàng từ màn hình Quản lý đơn hàng (ví dụ: để    |
| Flow**       | thông báo hết màu/size). Các bước gửi và nhận tin     |
|              | nhắn tiếp theo diễn ra tương tự như luồng Basic Flow. |
+--------------+-------------------------------------------------------+
| **Exception  | 2a. Lỗi mất kết nối mạng:                             |
| Flow**       |                                                       |
|              | 2a1. Hệ thống không thể duy trì phiên WebSockets. Hệ  |
|              | thống hiển thị thông báo \"Đang mất kết nối\...\" và  |
|              | tạm thời làm mờ nút \"Gửi\" để ngăn chặn mất dữ liệu. |
|              |                                                       |
|              | 2a2. Khi có mạng trở lại, hệ thống tự động kết nối    |
|              | (Re-connect) và fetch lại các tin nhắn bị nhỡ (nếu    |
|              | có).                                                  |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR13-1 (QĐ_KH13 / QĐ_SL6):** Cuộc trò chuyện     |
| Rules**      | mang tính chất riêng tư giữa một Khách hàng cụ thể và |
|              | Gian hàng cụ thể của sản phẩm đó.                     |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR13-1 (Cốt lõi):** Tính năng bắt buộc sử dụng  |
| n-Functional | giao thức **WebSockets** (Cụ thể là Spring            |
| R            | WebSockets + STOMP ở Back-end và thư viện             |
| equirement** | SockJS/stompjs ở Front-end) để duy trì kết nối liên   |
|              | tục thay vì dùng HTTP Requests thông thường. Độ trễ   |
|              | tin nhắn (Latency) phải đảm bảo dưới 1 giây.          |
+--------------+-------------------------------------------------------+

### Use case 14

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC14                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Hồ sơ và Gian hàng                            |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người bán, tôi muốn thiết lập thông tin doanh  |
| escription** | nghiệp, tài khoản ngân hàng và địa chỉ kho để hệ      |
|              | thống có thể đối soát dòng tiền và lấy hàng giao cho  |
|              | khách.                                                |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Người bán (Seller)                                    |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người bán truy cập vào tab \"Profile\" (Hồ sơ).       |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người bán đã đăng nhập thành công vào hệ thống.       |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Hồ sơ được cập nhật thành công, dữ liệu được ghi nhận |
| ndition(s)** | vào cơ sở dữ liệu.                                    |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người bán truy cập trang cấu hình Hồ sơ gian      |
| Flow**       | hàng.                                                 |
|              |                                                       |
|              | 2\. Hệ thống hiển thị Dashboard với các khối thông    |
|              | tin: Thông tin Doanh nghiệp (Business Details), Chi   |
|              | tiết Ngân hàng (Bank Details), và Địa chỉ Kho hàng    |
|              | (Pickup Address).                                     |
|              |                                                       |
|              | 3\. Người bán nhập hoặc chỉnh sửa các trường thông    |
|              | tin cần thiết (Tên Shop, Mã số thuế GST, Tên chủ tài  |
|              | khoản, Số tài khoản ngân hàng, Địa chỉ kho lấy hàng). |
|              |                                                       |
|              | 4\. (Tùy chọn) Người bán tải lên hình ảnh Logo và     |
|              | Banner cho gian hàng.                                 |
|              |                                                       |
|              | 5\. Người bán nhấn \"Lưu thay đổi\".                  |
|              |                                                       |
|              | 6\. Hệ thống kiểm tra dữ liệu, upload ảnh lên         |
|              | Cloudinary và lưu thông tin vào cơ sở dữ liệu.        |
|              |                                                       |
|              | 7\. Hệ thống thông báo \"Cập nhật thành công\" và làm |
|              | mới dữ liệu trên màn hình.                            |
+--------------+-------------------------------------------------------+
| *            | 7a. Xem trước gian hàng (Preview Storefront): Sau khi |
| *Alternative | lưu thành công ở bước 7, Người bán nhấn vào nút \"Xem |
| Flow**       | gian hàng\". Hệ thống mở một thẻ trình duyệt mới hiển |
|              | thị giao diện mặt tiền công khai của Shop (bao gồm    |
|              | Logo, Banner vừa đổi và danh sách các sản phẩm đang   |
|              | bán).).                                               |
+--------------+-------------------------------------------------------+
| **Exception  | 5a. Người bán bỏ trống các thông tin quan trọng để    |
| Flow**       | đối soát như Số tài khoản ngân hàng hoặc Mã số thuế   |
|              | (GST).                                                |
|              |                                                       |
|              | 5a1. Hệ thống báo lỗi validation ngay dưới trường     |
|              | nhập liệu và chặn lệnh lưu.                           |
|              |                                                       |
|              | *Use Case quay lại bước 3.*                           |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR14-1 (QĐ_SL1):** Seller bắt buộc phải cung cấp |
| Rules**      | Mã số thuế doanh nghiệp (GST), địa chỉ kho lấy hàng   |
|              | (Pickup address) và thông tin ngân hàng hợp lệ để hệ  |
|              | thống có cơ sở trả tiền đối soát (Payout) sau khi     |
|              | hoàn tất đơn hàng.                                    |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR14-1:** Các tệp tin Logo và Banner của Shop   |
| n-Functional | phải được đẩy trực tiếp lên Cloudinary API từ phía    |
| R            | Client (sử dụng FormData) để giảm tải băng thông cho  |
| equirement** | Server backend.                                       |
+--------------+-------------------------------------------------------+

### Use case 15

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC15                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Kho sản phẩm                                  |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người bán, tôi muốn đăng tải sản phẩm mới hoặc |
| escription** | cập nhật giá bán, số lượng tồn kho để hàng hóa có thể |
|              | được hiển thị tới Khách hàng.                         |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Người bán (Seller)                                    |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người bán chọn \"Add Product\" (Thêm sản phẩm) hoặc   |
|              | \"Update Stock\" (Cập nhật kho) tại trang Quản lý sản |
|              | phẩm.                                                 |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Tài khoản Người bán phải ở trạng thái được phê duyệt  |
| ndition(s)** | (ACTIVE).                                             |
+--------------+-------------------------------------------------------+
| **Post-Co    | Sản phẩm được lưu vào hệ thống và chuyển trạng thái   |
| ndition(s)** | thành chờ kiểm duyệt hoặc được hiển thị trực tiếp     |
|              | (tuỳ chính sách Admin).                               |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người bán chọn lệnh \"Thêm sản phẩm mới\".        |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu thông tin Sản phẩm.    |
|              |                                                       |
|              | 3\. Người bán chọn Danh mục phân cấp cho sản phẩm     |
|              | (Level 1, Level 2, Level 3).                          |
|              |                                                       |
|              | 4\. Người bán điền Tiền gốc (MRP) và Giá bán thực tế  |
|              | (Selling Price). Hệ thống tự động tính toán % Giảm    |
|              | giá dựa trên 2 mức giá này.                           |
|              |                                                       |
|              | 5\. Người bán cung cấp Tên sản phẩm, Mô tả, Kích cỡ,  |
|              | Màu sắc và Số lượng tồn kho (Quantity).               |
|              |                                                       |
|              | 6\. Người bán tải lên các hình ảnh thực tế của sản    |
|              | phẩm (tối đa 4-5 ảnh).                                |
|              |                                                       |
|              | 7\. Người bán nhấn lệnh \"Đăng sản phẩm\".            |
|              |                                                       |
|              | 8\. Hệ thống ghi nhận sản phẩm vào cơ sở dữ liệu. Sản |
|              | phẩm mặc định được đẩy vào trạng thái \"Chờ duyệt\"   |
|              | (Pending Approval).                                   |
+--------------+-------------------------------------------------------+
| *            | **1a. Cập nhật tồn kho (Update Stock):**              |
| *Alternative |                                                       |
| Flow**       | 1a1. Tại lưới danh sách Sản phẩm hiện có, Người bán   |
|              | chỉnh sửa trực tiếp con số tồn kho và nhấn icon       |
|              | \"Update\".                                           |
|              |                                                       |
|              | 1a2. Hệ thống cập nhật nhanh tồn kho hiện tại         |
|              | (In-stock) và hiển thị lại lưới mà không cần tải      |
|              | trang.                                                |
+--------------+-------------------------------------------------------+
| **Exception  | **4a.** Người bán điền Giá bán thực tế (Selling       |
| Flow**       | Price) LỚN HƠN Giá gốc (MRP).                         |
|              |                                                       |
|              | **4a1.** Hệ thống báo lỗi logic \"Giá bán không được  |
|              | lớn hơn giá gốc\" và chặn nút lưu.                    |
|              |                                                       |
|              | **6a.** Quá trình upload ảnh lên máy chủ Cloudinary   |
|              | thất bại do lỗi mạng.\<br\>**6a1.** Vòng xoay tiến    |
|              | trình (Circular Progress) báo lỗi và yêu cầu Người    |
|              | bán chọn lại ảnh.                                     |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR15-1 (QĐ_SL2):** Mỗi sản phẩm bắt buộc phải    |
| Rules**      | được gắn vào đúng 1 Cây danh mục Level 3 (Ví dụ: Men  |
|              | -\> Topwear -\> T-Shirt). Khách hàng sẽ dùng ID danh  |
|              | mục này để lọc sản phẩm tại Trang chủ (UC01).         |
|              |                                                       |
|              | \- **BR15-2 (QĐ_AD11):** Để bảo vệ nền tảng, mọi sản  |
|              | phẩm tạo mới đều ở trạng thái ẩn (PENDING). Chỉ khi   |
|              | Admin (Quản trị viên) phê duyệt ở UC20, sản phẩm mới  |
|              | được hiển thị công khai trên gian hàng.               |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR15-1:** Khi người bán thay đổi giá tiền ở     |
| n-Functional | bước 4, % Giảm giá phải được tự động tính toán bằng   |
| R            | JavaScript trên giao diện (Client-side) ngay lập tức  |
| equirement** | theo thời gian thực.                                  |
+--------------+-------------------------------------------------------+

### Use case 16

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC16                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Xử lý Đơn hàng & Vận chuyển                           |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người bán, tôi muốn tiếp nhận đơn đặt hàng,    |
| escription** | xác nhận đóng gói và kết xuất mã vận đơn để tiến hành |
|              | giao hàng cho khách.                                  |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Người bán (Seller)                                    |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người bán truy cập vào tab \"Orders\" (Đơn hàng) để   |
|              | xem các đơn hàng vừa được khách đặt.                  |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Có đơn hàng phát sinh thuộc về mã gian hàng của       |
| ndition(s)** | Seller đó.                                            |
+--------------+-------------------------------------------------------+
| **Post-Co    | Trạng thái đơn hàng thay đổi, Mã vận đơn được sinh ra |
| ndition(s)** | để hai bên cùng tra cứu.                              |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người bán truy cập tab Quản lý đơn hàng. Hệ thống |
| Flow**       | hiển thị danh sách các đơn hàng theo lưới (Có thể     |
|              | phân bộ lọc: Mới đặt, Chờ lấy hàng, Đang giao\...).   |
|              |                                                       |
|              | 2\. Người bán nhấn vào một đơn hàng Mới (Trạng thái:  |
|              | PLACED) và chọn hành động \"Xác nhận đơn\" (Confirm). |
|              |                                                       |
|              | 3\. Trạng thái đơn hàng chuyển sang CONFIRMED (Đã xác |
|              | nhận).                                                |
|              |                                                       |
|              | 4\. Người bán tiến hành đóng gói, sau đó chọn lệnh    |
|              | \"Đẩy đơn vận chuyển\" (Ship Order).                  |
|              |                                                       |
|              | 5\. Hệ thống Back-end tự động gọi API giao tiếp với   |
|              | đối tác vận chuyển (VD: GHTK, Grab) truyền đi thông   |
|              | tin Khách hàng và Địa chỉ kho Seller (Pickup          |
|              | Address).                                             |
|              |                                                       |
|              | 6\. API vận chuyển trả về Mã vận đơn (Tracking ID).   |
|              | Hệ thống lưu mã này vào Đơn hàng, đổi trạng thái      |
|              | thành SHIPPED (Đang giao).                            |
|              |                                                       |
|              | 7\. Người bán chọn lệnh \"In phiếu giao hàng\", hệ    |
|              | thống tự động kết xuất (Generate) tài liệu PDF mã     |
|              | vạch để Seller dán lên gói hàng.                      |
+--------------+-------------------------------------------------------+
| *            | 6a. Cập nhật qua Webhook: Sau khi hàng được đẩy đi ở  |
| *Alternative | bước 6, người bán không cần tác động thủ công nữa.    |
| Flow**       | Khi Shipper giao hàng thành công, hệ thống của ĐVVC   |
|              | sẽ tự bắn API (Webhook) về máy chủ E-commerce để đổi  |
|              | trạng thái đơn hàng sang DELIVERED (Đã giao).         |
+--------------+-------------------------------------------------------+
| **Exception  | 5a. API kết nối với Đơn vị vận chuyển gặp sự cố       |
| Flow**       | (Timeout) hoặc Địa chỉ kho lấy hàng của Seller không  |
|              | hợp lệ.                                               |
|              |                                                       |
|              | 5a1. Hệ thống báo lỗi \"Kết nối hãng vận chuyển thất  |
|              | bại. Vui lòng thử lại sau\" và giữ nguyên trạng thái  |
|              | đơn ở mức CONFIRMED.                                  |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR16-1 (QĐ_SL3):** Việc kết nối tạo mã vận đơn   |
| Rules**      | (Tracking ID) là quy trình xử lý tự động              |
|              | (Automation). Seller không được tự nhập tay mã        |
|              | tracking để tránh gian lận đối soát phí ship.         |
|              |                                                       |
|              | \- **BR16-2 (QĐ_SL4):** Phiếu giao hàng (Vận đơn PDF) |
|              | bắt buộc chứa Barcode/QR Code của ĐVVC để tài xế có   |
|              | thể dùng máy quét tít mã lấy hàng.                    |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR16-1:** Vì lệnh gọi API ra bên thứ 3 (Hãng    |
| n-Functional | vận chuyển) có độ trễ, giao diện phải hiển thị        |
| R            | Loading Spinner chặn thao tác tay của Seller tránh    |
| equirement** | việc ấn \"Đẩy đơn\" liên tục sinh ra nhiều mã vận đơn |
|              | rác.                                                  |
+--------------+-------------------------------------------------------+

### Use case 17

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC17                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Xử lý Yêu cầu Hoàn trả                                |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người bán, tôi muốn xem xét lý do và các minh  |
| escription** | chứng để đưa ra quyết định chấp nhận hoặc từ chối yêu |
|              | cầu trả hàng/hoàn tiền từ Khách hàng.                 |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Người bán (Seller)                                    |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người bán nhận được thông báo hoặc truy cập trực tiếp |
|              | vào tab \"Yêu cầu hoàn trả\" (Return Requests).       |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Có ít nhất một đơn hàng đang ở trạng thái yêu cầu     |
| ndition(s)** | hoàn trả (RETURN_REQUESTED).                          |
+--------------+-------------------------------------------------------+
| **Post-Co    | Yêu cầu được giải quyết. Trạng thái đơn hàng được cập |
| ndition(s)** | nhật và hệ thống tự động xử lý các bước hoàn tiền/đối |
|              | soát tiếp theo.                                       |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người bán truy cập danh sách các yêu cầu trả hàng |
| Flow**       | từ Khách hàng.                                        |
|              |                                                       |
|              | 2\. Hệ thống truy xuất và hiển thị danh sách các đơn  |
|              | hàng đang ở trạng thái RETURN_REQUESTED.              |
|              |                                                       |
|              | 3\. Người bán chọn xem chi tiết một yêu cầu, đọc lý   |
|              | do và xem các hình ảnh/video minh chứng do Khách hàng |
|              | cung cấp.                                             |
|              |                                                       |
|              | 4\. Người bán quyết định chọn lệnh **\"**Chấp nhận\"  |
|              | (Accept).                                             |
|              |                                                       |
|              | 5\. Hệ thống gửi thông báo cho Khách hàng yêu cầu gửi |
|              | trả hàng về kho.                                      |
|              |                                                       |
|              | 6\. Sau khi Người bán nhận lại hàng và bấm xác nhận,  |
|              | hệ thống tự động gọi API hoàn tiền trả về tài khoản   |
|              | Khách hàng và cập nhật đơn hàng thành trạng thái      |
|              | REFUNDED.                                             |
+--------------+-------------------------------------------------------+
| *            | 4a. Người bán chọn lệnh \"Từ chối\" (Reject):         |
| *Alternative |                                                       |
| Flow**       | 4a1. Người bán bấm \"Từ chối\" đối với yêu cầu của    |
|              | Khách hàng.                                           |
|              |                                                       |
|              | 4a2. Hệ thống hiển thị khung nhập liệu yêu cầu nhập   |
|              | lý do từ chối.                                        |
|              |                                                       |
|              | 4a3. Người bán điền lý do và xác nhận.                |
|              |                                                       |
|              | 4a4. Hệ thống ghi nhận quyết định, thông báo cho      |
|              | Khách hàng lý do từ chối và mở ra khả năng cho phép   |
|              | Khách hàng Khiếu nại (Dispute) lên Admin.             |
+--------------+-------------------------------------------------------+
| **Exception  | 1a. Vượt quá thời hạn xử lý:                          |
| Flow**       |                                                       |
|              | 1a1. Quá thời hạn 3 ngày kể từ khi Khách hàng tạo yêu |
|              | cầu mà Người bán chưa có phản hồi, hệ thống hiển thị  |
|              | cảnh báo vi phạm thời gian xử lý (SLA) đối với gian   |
|              | hàng đó.                                              |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR17-1 (QĐ_SL5):** Seller có tối đa 3 ngày để    |
| Rules**      | phản hồi yêu cầu. Trong trường hợp Từ chối, Seller    |
|              | bắt buộc phải ghi rõ lý do từ chối để Khách hàng (và  |
|              | Admin sau này) có cơ sở đối chứng.                    |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR17-1:** Giao diện xem hình ảnh/video minh     |
| n-Functional | chứng tải từ Cloudinary phải hỗ trợ Zoom (phóng to)   |
| R            | và phát media mượt mà trực tiếp trên Dashboard mà     |
| equirement** | không cần tải file về máy.                            |
+--------------+-------------------------------------------------------+

### Use case 18

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC18                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Theo dõi Đối soát & Doanh thu                         |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người bán, tôi muốn xem biểu đồ doanh thu tổng |
| escription** | quan, theo dõi lịch sử dòng tiền và xuất dữ liệu ra   |
|              | file Excel để thực hiện nghiệp vụ kế toán nội bộ.     |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Người bán (Seller)                                    |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người bán truy cập vào trang \"Bảng điều khiển\"      |
|              | (Dashboard) hoặc tab \"Giao dịch\" (Transactions).    |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người bán đã đăng nhập thành công vào hệ thống.       |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Dữ liệu thống kê được tính toán chính xác và kết xuất |
| ndition(s)** | thành file báo cáo thành công.                        |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người bán truy cập trang Bảng điều khiển tổng     |
| Flow**       | quan.                                                 |
|              |                                                       |
|              | 2\. Hệ thống tự động tổng hợp và tính toán các chỉ số |
|              | kinh doanh theo thời gian thực.                       |
|              |                                                       |
|              | 3\. Hệ thống hiển thị các thẻ thống kê tổng thể bao   |
|              | gồm: Tổng thu nhập (Total Earning), Tổng số đơn       |
|              | (Total Orders), Đơn bị hủy (Canceled Orders) và Tổng  |
|              | hoàn tiền (Total Refund).                             |
|              |                                                       |
|              | 4\. Hệ thống hiển thị Biểu đồ doanh thu trực quan     |
|              | (Earning graphs) phân bổ theo ngày, tuần, hoặc tháng. |
|              |                                                       |
|              | 5\. Người bán truy cập tab \"Lịch sử giao dịch\"      |
|              | (Transactions) để xem đối soát chi tiết dòng tiền của |
|              | từng đơn hàng cụ thể.                                 |
+--------------+-------------------------------------------------------+
| *            | 5a. Xuất báo cáo dữ liệu Excel:                       |
| *Alternative |                                                       |
| Flow**       | 5a1. Tại màn hình Giao dịch/Báo cáo, Người bán chọn   |
|              | lệnh \"Xuất báo cáo\" (Export).                       |
|              |                                                       |
|              | 5a2. Hệ thống truy xuất dữ liệu từ Database, định     |
|              | dạng lại thành cấu trúc tệp tin .xlsx                 |
|              |                                                       |
|              | 5a3. Hệ thống tự động tải file báo cáo Excel xuống    |
|              | thiết bị của Người bán.                               |
+--------------+-------------------------------------------------------+
| **Exception  | 2a. Hệ thống chưa phát sinh giao dịch:                |
| Flow**       |                                                       |
|              | 2a1. Nếu Seller là gian hàng mới chưa có đơn hàng     |
|              | nào, hệ thống không báo lỗi mà chỉ hiển thị dữ liệu   |
|              | \"0\" trên các thẻ và hiển thị trạng thái \"Empty     |
|              | state\" (Chưa có dữ liệu) tại biểu đồ đồ thị.         |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR18-1 (CT_SL1):** Báo cáo phải tự động tổng hợp |
| Rules**      | chính xác các chỉ số từ tất cả các đơn hàng thuộc     |
|              | quyền sở hữu của Seller.                              |
|              |                                                       |
|              | \- **BR18-2 (CT_SL2):** Khi một đơn hàng hoàn tiền    |
|              | thành công ở UC17, khoản tiền này phải được tự động   |
|              | trừ khỏi Tổng thu nhập (Total Earning) và được cộng   |
|              | dồn vào thống kê Tổng hoàn tiền (Total Refund) để     |
|              | việc đối soát dòng tiền luôn minh bạch.               |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR18-1 (QĐ_SL7):** File Excel được kết xuất     |
| n-Functional | phải đảm bảo đúng định dạng bảng tính (Spreadsheet),  |
| R            | không bị lỗi font chữ Unicode (tiếng Việt) để Người   |
| equirement** | bán dễ dàng làm việc với các phần mềm kế toán.        |
+--------------+-------------------------------------------------------+

### Use case 19

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC19                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Kiểm duyệt Người bán                                  |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn xét duyệt hồ sơ đăng ký    |
| escription** | gian hàng hoặc xử lý vi phạm để duy trì môi trường    |
|              | kinh doanh minh bạch và chất lượng trên sàn.          |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Sellers\" trên Admin         |
|              | Dashboard.                                            |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Trạng thái của tài khoản Seller được thay đổi         |
| ndition(s)** | (ACTIVE, SUSPENDED, hoặc BANNED).                     |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin truy cập danh sách quản lý Người bán.       |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị lưới danh sách, Admin lọc theo  |
|              | trạng thái PENDING_VERIFICATION (Chờ xác minh).       |
|              |                                                       |
|              | 3\. Admin nhấp vào xem chi tiết hồ sơ của một Người   |
|              | bán (bao gồm thông tin GST, địa chỉ kho, số tài khoản |
|              | ngân hàng).                                           |
|              |                                                       |
|              | 4\. Admin kiểm tra tính hợp lệ của thông tin và chọn  |
|              | lệnh \"Approve\" (Phê duyệt).                         |
|              |                                                       |
|              | 5\. Hệ thống thay đổi trạng thái tài khoản thành      |
|              | ACTIVE.                                               |
|              |                                                       |
|              | 6\. Hệ thống tự động gửi Email thông báo chúc mừng    |
|              | đến Người bán và cho phép họ bắt đầu đăng sản phẩm.   |
+--------------+-------------------------------------------------------+
| *            | 4a. Quản lý vi phạm (Đình chỉ / Cấm vĩnh viễn):       |
| *Alternative |                                                       |
| Flow**       | 4a1. Tại danh sách các Seller đang hoạt động          |
|              | (ACTIVE), Admin phát hiện gian hàng vi phạm chính     |
|              | sách.                                                 |
|              |                                                       |
|              | 4a2. Admin chọn lệnh đổi trạng thái sang Suspend      |
|              | (Đình chỉ tạm thời) hoặc Ban (Cấm vĩnh viễn).         |
|              |                                                       |
|              | 4a3. Hệ thống khóa quyền đăng nhập của Seller đó và   |
|              | ẩn toàn bộ sản phẩm của gian hàng khỏi Trang chủ.     |
+--------------+-------------------------------------------------------+
| **Exception  | 4b. Hồ sơ Seller đăng ký thiếu các thông tin cốt lõi  |
| Flow**       | (Không có GST hoặc sai định dạng tài khoản ngân       |
|              | hàng).                                                |
|              |                                                       |
|              | 4b1. Admin chọn lệnh \"Reject\" (Từ chối), ghi rõ lý  |
|              | do. Hệ thống gửi email yêu cầu Seller bổ sung thông   |
|              | tin.                                                  |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR19-1 (QĐ_AD1):** Chỉ những Seller có trạng     |
| Rules**      | thái ACTIVE mới được quyền truy cập vào Seller        |
|              | Dashboard và đăng tải sản phẩm.                       |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR19-1:** Mọi thao tác thay đổi trạng thái tài  |
| n-Functional | khoản Seller đều phải được lưu vết lại vào bảng Audit |
| R            | Log (Lịch sử hệ thống) để phục vụ kiểm toán sau này.  |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 20

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC20                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Kiểm duyệt Sản phẩm                                   |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn xem xét thông tin và hình  |
| escription** | ảnh của các sản phẩm mới do Seller đăng tải để phê    |
|              | duyệt hiển thị công khai hoặc từ chối.                |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Kiểm duyệt Sản phẩm\"        |
|              | (Pending Products).                                   |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Có ít nhất một sản phẩm mới được Seller đăng tải đang |
| ndition(s)** | ở trạng thái PENDING.                                 |
+--------------+-------------------------------------------------------+
| **Post-Co    | Sản phẩm được phê duyệt hiển thị lên sàn hoặc bị trả  |
| ndition(s)** | về cho Seller.                                        |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin mở danh sách các sản phẩm đang chờ duyệt.   |
| Flow**       |                                                       |
|              | 2\. Admin chọn một sản phẩm và kiểm tra chi tiết:     |
|              | Tên, Mô tả, Cây danh mục, Giá bán, và Hình ảnh thực   |
|              | tế.                                                   |
|              |                                                       |
|              | 3\. Admin xác nhận sản phẩm không vi phạm quy định    |
|              | (không phải hàng giả, hàng cấm) và nhấn nút           |
|              | \"Approve\" (Phê duyệt).                              |
|              |                                                       |
|              | 4\. Hệ thống chuyển trạng thái sản phẩm sang          |
|              | PUBLISHED (Đã xuất bản).                              |
|              |                                                       |
|              | 5\. Sản phẩm ngay lập tức xuất hiện trên giao diện    |
|              | tìm kiếm và mặt tiền (Storefront) của Khách hàng.     |
+--------------+-------------------------------------------------------+
| *            | 3a. Admin Từ chối sản phẩm:                           |
| *Alternative |                                                       |
| Flow**       | 3a1. Admin phát hiện sản phẩm sai danh mục hoặc ảnh   |
|              | kém chất lượng, chọn lệnh \"Reject\" (Từ chối).       |
|              |                                                       |
|              | 3a2. Hệ thống yêu cầu nhập lý do từ chối. Admin điền  |
|              | lý do và xác nhận.                                    |
|              |                                                       |
|              | 3a3. Sản phẩm bị đẩy về trạng thái REJECTED, Seller   |
|              | nhận được thông báo để chỉnh sửa lại.                 |
+--------------+-------------------------------------------------------+
| **Exception  | *Không có nhánh ngoại lệ phức tạp.*                   |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR20-1 (QĐ_AD11):** Bắt buộc áp dụng cơ chế kiểm |
| Rules**      | duyệt. Hệ thống tự động thiết lập trạng thái mặc định |
|              | của mọi sản phẩm tạo mới là PENDING. Seller không thể |
|              | tự ý Publish sản phẩm.                                |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR20-1:** Giao diện duyệt sản phẩm phải được    |
| n-Functional | thiết kế tối ưu (Load ảnh nhanh từ Cloudinary) để     |
| R            | Admin có thể duyệt hàng loạt (Mass approval) một cách |
| equirement** | nhanh chóng.                                          |
+--------------+-------------------------------------------------------+

### Use case 21

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC21                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Giao diện Trang chủ                           |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn cấu hình động các Banner,  |
| escription** | Deals và lưới danh mục (Grid Categories) trên Trang   |
|              | chủ mà không cần phải nhờ Lập trình viên sửa code.    |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Home Page Config\" trên      |
|              | Dashboard.                                            |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Giao diện trang chủ (Homepage) của Khách hàng được tự |
| ndition(s)** | động cập nhật layout mới nhất.                        |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin truy cập module quản lý Trang chủ.          |
| Flow**       |                                                       |
|              | 2\. Hệ thống hiển thị các Section có thể tùy biến     |
|              | (Electric Categories, Grid Categories, Shop by        |
|              | Category).                                            |
|              |                                                       |
|              | 3\. Admin chọn chỉnh sửa một Section (Ví dụ: Lưới     |
|              | danh mục).                                            |
|              |                                                       |
|              | 4\. Admin cung cấp đường dẫn hình ảnh mới (Image      |
|              | URL), chọn Danh mục cấp 3 tương ứng (Ví dụ: Women -\> |
|              | Footwear -\> Heels).                                  |
|              |                                                       |
|              | 5\. Admin nhấn \"Cập nhật\".                          |
|              |                                                       |
|              | 6\. Hệ thống lưu cấu hình JSON vào cơ sở dữ liệu và   |
|              | thông báo cập nhật thành công.                        |
|              |                                                       |
|              | 7\. Admin tải lại Trang chủ Client, hình ảnh và liên  |
|              | kết danh mục mới ngay lập tức được áp dụng.           |
+--------------+-------------------------------------------------------+
| *            | *Không có. Thao tác được xử lý trực tiếp trên biểu    |
| *Alternative | mẫu cấu hình (Form-based).*                           |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | **4a.** Admin nhập URL hình ảnh sai định dạng hoặc bị |
| Flow**       | lỗi.                                                  |
|              |                                                       |
|              | 4a1. Hình ảnh xem trước (Preview) bị vỡ. Hệ thống báo |
|              | lỗi \"URL hình ảnh không hợp lệ\".                    |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR21-1 (QĐ_AD3):** Dữ liệu cấu hình trang chủ    |
| Rules**      | (Homepage Data) phải được thiết kế thành một API      |
|              | public độc lập, để Client App (React) có thể gọi và   |
|              | render giao diện tự động dựa trên cấu hình mà Admin   |
|              | vừa lưu.                                              |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR21-1:** Trải nghiệm thay đổi giao diện theo   |
| n-Functional | nguyên tắc WYSIWYG (What You See Is What You Get).    |
| R            | Các thay đổi cấu hình phải phản hồi tức thì dưới 1    |
| equirement** | giây.                                                 |
+--------------+-------------------------------------------------------+

### Use case 22

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC22                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Chiến dịch Khuyến mãi                         |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn phát hành và quản lý các   |
| escription** | Mã giảm giá (Coupons) và Khuyến mãi danh mục (Deals)  |
|              | để kích thích nhu cầu mua sắm chung trên toàn sàn.    |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập menu \"Coupons\" hoặc \"Deals\".       |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Coupon hoặc Deal mới được phát hành, có sẵn để Khách  |
| ndition(s)** | hàng áp dụng.                                         |
+--------------+-------------------------------------------------------+
| **Basic      | Trường hợp Tạo Coupon:                                |
| Flow**       |                                                       |
|              | 1\. Admin chọn lệnh \"Tạo Coupon mới\" (Add New       |
|              | Coupon).                                              |
|              |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu khởi tạo.              |
|              |                                                       |
|              | 3\. Admin điền Mã Coupon (VD: DIWALI), % giảm giá     |
|              | (VD: 10%), Ngày bắt đầu, Ngày kết thúc và Giá trị đơn |
|              | hàng tối thiểu (Min Order Value).                     |
|              |                                                       |
|              | 4\. Admin nhấn \"Khởi tạo\".                          |
|              |                                                       |
|              | 5\. Hệ thống lưu Coupon vào hệ thống và kích hoạt     |
|              | trạng thái (Active).                                  |
+--------------+-------------------------------------------------------+
| *            | 1a. Cập nhật Deal giảm giá:                           |
| *Alternative |                                                       |
| Flow**       | 1a1. Admin chuyển sang tab \"Deals\", chọn một Deal   |
|              | hiện có (VD: Thời trang Nữ giảm 80%).                 |
|              |                                                       |
|              | 1a2. Admin thay đổi phần trăm giảm giá hoặc đổi ảnh   |
|              | banner của Deal đó.                                   |
|              |                                                       |
|              | 1a3. Hệ thống lưu lại và làm mới (refresh) khu vực    |
|              | Today\'s Deal trên Trang chủ của người dùng.          |
+--------------+-------------------------------------------------------+
| **Exception  | 3a. Admin chọn Ngày kết thúc (Validity End Date) diễn |
| Flow**       | ra TRƯỚC Ngày bắt đầu (Validity Start Date).          |
|              |                                                       |
|              | 3a1. Hệ thống báo lỗi logic thời gian và chặn thao    |
|              | tác lưu.                                              |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR22-1 (QĐ_AD4):** Các chiến dịch do Admin tạo   |
| Rules**      | áp dụng chung cho toàn nền tảng. Khi Khách hàng áp    |
|              | dụng Coupon này ở UC03/UC05, hệ thống sẽ đối soát để  |
|              | đảm bảo Seller không bị lỗ doanh thu.                 |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR22-1:** Quá trình kiểm tra Coupon tại giỏ     |
| n-Functional | hàng của khách hàng đối với các Coupon do Admin tạo   |
| R            | phải được truy vấn với tốc độ dưới 100ms.             |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 23

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC23                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Giải quyết khiếu nại (Disputes)                       |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn làm trọng tài xử lý các    |
| escription** | đơn hàng đang có tranh chấp giữa Khách hàng và Người  |
|              | bán, đưa ra phán quyết cuối cùng và gọi lệnh hoàn     |
|              | tiền.                                                 |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Giải quyết Khiếu nại\"       |
|              | (Disputes).                                           |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Có đơn hàng đang bị khóa ở trạng thái tranh chấp      |
| ndition(s)** | (DISPUTED) (kết quả từ UC09).                         |
+--------------+-------------------------------------------------------+
| **Post-Co    | Đơn hàng được giải quyết xong, tiền được hoàn cho     |
| ndition(s)** | khách hoặc đẩy cho Seller.                            |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin xem danh sách các đơn hàng DISPUTED.        |
| Flow**       |                                                       |
|              | 2\. Admin chọn một đơn hàng, hệ thống hiển thị song   |
|              | song 2 luồng dữ liệu: Lý do từ chối của Seller và     |
|              | Bằng chứng (Video/Ảnh) của Khách hàng.                |
|              |                                                       |
|              | 3\. Admin đóng vai trò trọng tài để đánh giá minh     |
|              | chứng.                                                |
|              |                                                       |
|              | 4\. Nếu lỗi thuộc về Seller, Admin chọn lệnh \"Chấp   |
|              | nhận khiếu nại - Hoàn tiền\".                         |
|              |                                                       |
|              | 5\. Hệ thống Backend tự động gọi API Refund của cổng  |
|              | thanh toán điện tử (VnPay/SePay/Momo) để trả lại tiền |
|              | về thẻ của khách hàng.                                |
|              |                                                       |
|              | 6\. Trạng thái đơn hàng chuyển thành REFUNDED và      |
|              | khiếu nại đóng lại.                                   |
+--------------+-------------------------------------------------------+
| *            | 4a. Phán quyết bảo vệ Người bán:                      |
| *Alternative |                                                       |
| Flow**       | 4a1. Admin nhận thấy bằng chứng của Khách hàng không  |
|              | hợp lý hoặc có dấu hiệu gian lận.                     |
|              |                                                       |
|              | 4a2. Admin chọn lệnh \"Từ chối khiếu nại\".           |
|              |                                                       |
|              | 4a3. Hệ thống đóng khiếu nại, mở khóa dòng tiền đối   |
|              | soát và chuyển doanh thu của đơn hàng đó vào ví của   |
|              | Seller.                                               |
+--------------+-------------------------------------------------------+
| **Exception  | 5a. Lỗi API Cổng thanh toán:                          |
| Flow**       |                                                       |
|              | 5a1. Hệ thống Backend gọi API Refund nhưng cổng       |
|              | Stripe/VnPay bị timeout hoặc từ chối lệnh.            |
|              |                                                       |
|              | 5a2. Hệ thống báo lỗi cho Admin \"Hoàn tiền thất bại  |
|              | do lỗi cổng thanh toán\", giữ nguyên trạng thái       |
|              | DISPUTED để Admin thử lại sau.                        |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR23-1 (QĐ_AD6):** Phán quyết của Admin là quyết |
| Rules**      | định cao nhất và bắt buộc thực thi. Sau khi Admin đã  |
|              | xử lý, cả Khách hàng và Seller không được phép thao   |
|              | tác khiếu nại lại đối với đơn hàng này.               |
|              |                                                       |
|              | \- **BR23-2 (QĐ_AD7):** Tuyệt đối không hoàn tiền thủ |
|              | công. Việc trả tiền phải thông qua luồng tự động      |
|              | (Automated Refund API) để đảm bảo an toàn dòng tiền   |
|              | hệ thống.                                             |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR23-1:** Tính năng so sánh minh chứng          |
| n-Functional | (Evidences) phải cho phép tải và phát video trực tiếp |
| R            | trên dashboard của Admin mà không cần redirect qua    |
| equirement** | trang web khác.                                       |
+--------------+-------------------------------------------------------+

### Use case 24

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC24                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Quản lý Khách hàng                                    |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn xem danh sách khách hàng   |
| escription** | và quản lý trạng thái tài khoản để ngăn chặn các      |
|              | người dùng có hành vi vi phạm chính sách của nền      |
|              | tảng.                                                 |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Khách hàng\" (Customers)     |
|              | trên Admin Dashboard.                                 |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Trạng thái tài khoản khách hàng được cập nhật thành   |
| ndition(s)** | công.                                                 |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin chọn tab \"Khách hàng\" trên trình đơn điều |
| Flow**       | hướng.                                                |
|              |                                                       |
|              | 2\. Hệ thống tải và hiển thị lưới danh sách toàn bộ   |
|              | khách hàng trên hệ thống.                             |
|              |                                                       |
|              | 3\. Admin tìm kiếm hoặc chọn một khách hàng cụ thể để |
|              | xem chi tiết hồ sơ (Tên, Email, SĐT, Địa chỉ).        |
|              |                                                       |
|              | 4\. (Tùy chọn) Nếu phát hiện tài khoản có hành vi     |
|              | gian lận (Ví dụ: lạm dụng mã giảm giá, boom hàng      |
|              | nhiều lần), Admin chọn lệnh \"Khóa tài khoản\" (Ban   |
|              | Account).                                             |
|              |                                                       |
|              | 5\. Hệ thống xác nhận và thay đổi trạng thái tài      |
|              | khoản khách hàng thành BANNED.                        |
|              |                                                       |
|              | 6\. Hệ thống hiển thị thông báo \"Cập nhật trạng thái |
|              | thành công\" và khóa quyền đăng nhập của người dùng   |
|              | này.                                                  |
+--------------+-------------------------------------------------------+
| *            | *Không có nhánh rẽ phức tạp, Admin thao tác trực tiếp |
| *Alternative | trên lưới dữ liệu.*                                   |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | *Không có.*                                           |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR24-1 (QĐ_AD5):** Admin có quyền xem thông tin  |
| Rules**      | cơ bản của Khách hàng để hỗ trợ giải quyết sự cố,     |
|              | nhưng tuyệt đối không được xem mật khẩu của người     |
|              | dùng (mật khẩu đã được mã hóa BCrypt).                |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR24-1:** Bắt buộc áp dụng cơ chế Phân trang    |
| n-Functional | (Pagination) từ API Backend cho danh sách khách hàng  |
| R            | để đảm bảo hiệu suất khi dữ liệu có hàng triệu user.  |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 25

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC25                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Cấu hình Thông số Tài chính                           |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn thiết lập tỉ lệ quy đổi Ví |
| escription** | Xu và mức phí nền tảng (Platform Fee) để điều tiết    |
|              | chính sách kinh doanh và lợi nhuận của sàn.           |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Cấu hình Tài chính\" hoặc    |
|              | \"Cài đặt Hệ thống\" trên Dashboard.                  |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Các thông số tài chính mới được lưu lại và áp dụng    |
| ndition(s)** | ngay lập tức cho các giao dịch thanh toán và đối soát |
|              | tiếp theo.                                            |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin truy cập màn hình Cấu hình Thông số Tài     |
| Flow**       | chính.                                                |
|              |                                                       |
|              | 2\. Hệ thống hiển thị biểu mẫu bao gồm 2 nhóm cấu     |
|              | hình chính:                                           |
|              |                                                       |
|              |    - Cấu hình Ví Xu (Reward Coins): Tỉ lệ kiếm xu, Tỉ |
|              | giá tiêu xu, Hạn mức thanh toán bằng xu.              |
|              |                                                       |
|              |    - Cấu hình Phí sàn (Platform Fee): Phần trăm (%)   |
|              | phí trích xuất từ doanh thu mỗi đơn hàng thành công   |
|              | của Seller.                                           |
|              |                                                       |
|              | 3\. Admin nhập các thông số mới (Ví dụ: Thu phí nền   |
|              | tảng 5%).                                             |
|              |                                                       |
|              | 4\. Admin nhấn lệnh \"Lưu cấu hình\".                 |
|              |                                                       |
|              | 5\. Hệ thống ghi nhận các hằng số này vào cơ sở dữ    |
|              | liệu.                                                 |
|              |                                                       |
|              | 6\. Hệ thống hiển thị thông báo cập nhật thành công.  |
+--------------+-------------------------------------------------------+
| *            | *Không có.*                                           |
| *Alternative |                                                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | **3a.** Admin nhập số âm cho các trường Tỉ lệ phần    |
| Flow**       | trăm hoặc Tỉ giá quy đổi.                             |
|              |                                                       |
|              | 3a1. Hệ thống báo lỗi bôi đỏ \"Giá trị không hợp lệ,  |
|              | phải lớn hơn hoặc bằng 0\" và chặn lệnh lưu.          |
|              |                                                       |
|              | *Use Case quay lại bước 3.*                           |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR25-1 (QĐ_AD8):** Cấu hình xu phải đảm bảo 3    |
| Rules**      | thông số: Tỉ lệ kiếm (VD: 1000đ = 1 Xu), Tỉ giá tiêu  |
|              | (1 Xu = 1đ), và Hạn mức tối đa (VD: Xu chỉ thanh toán |
|              | tối đa 50% giá trị hóa đơn).                          |
|              |                                                       |
|              | \- **BR25-2 (QĐ_AD12):** % Phí nền tảng là biến cấu   |
|              | hình động, không fix cứng trong code. Thông số này sẽ |
|              | được gọi ra để áp dụng vào Công thức CT_AD1 khi đối   |
|              | soát doanh thu thực nhận cho Seller ở UC18.           |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR25-1:** Các thay đổi về thông số tài chính    |
| n-Functional | chỉ áp dụng cho các Đơn hàng (Orders) được tạo ra SAU |
| R            | thời điểm cập nhật. Các đơn hàng trong quá khứ tuyệt  |
| equirement** | đối không bị tính lại (để bảo toàn tính toàn vẹn của  |
|              | lịch sử kế toán).                                     |
+--------------+-------------------------------------------------------+

### Use case 26

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC26                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Theo dõi Nhật ký (Audit Log)                          |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là Quản trị viên, tôi muốn truy xuất và xem lại lịch  |
| escription** | sử các thao tác thay đổi dữ liệu quan trọng trên hệ   |
|              | thống để phục vụ công tác kiểm toán và rà soát lỗi.   |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Quản trị viên (Admin)                                 |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Admin truy cập vào tab \"Nhật ký hệ thống\" (Audit    |
|              | Logs).                                                |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Admin đã đăng nhập thành công.                        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Admin xem được toàn bộ lịch sử các thao tác quản trị  |
| ndition(s)** | một cách minh bạch.                                   |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Admin truy cập vào trang Nhật ký hệ thống.        |
| Flow**       |                                                       |
|              | 2\. Hệ thống truy xuất dữ liệu từ cơ sở dữ liệu và    |
|              | hiển thị danh sách các bản ghi nhật ký.               |
|              |                                                       |
|              | 3\. Mỗi bản ghi log bao gồm các thông tin: Thời gian  |
|              | thực hiện (Timestamp), Tên tài khoản thực hiện        |
|              | (Actor), Hành động (Action - VD: Ban Seller, Phê      |
|              | duyệt sản phẩm, Đổi phí nền tảng).                    |
|              |                                                       |
|              | 4\. Admin sử dụng bộ lọc để tìm kiếm các hành động cụ |
|              | thể theo khoảng thời gian hoặc theo phân hệ module.   |
|              |                                                       |
|              | 5\. Admin tra cứu thông tin phục vụ kiểm toán. Thao   |
|              | tác xem hoàn tất.                                     |
+--------------+-------------------------------------------------------+
| *            | *Không có.*                                           |
| *Alternative |                                                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | *Không có nhánh ngoại lệ phức tạp, chủ yếu xử lý lỗi  |
| Flow**       | timeout nếu dữ liệu quá lớn.*                         |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR26-1:** Bảng dữ liệu Nhật ký là dữ liệu Tuyệt  |
| Rules**      | đối Chỉ đọc (Read-only). Hệ thống không cung cấp chức |
|              | năng Xóa (Delete) hay Sửa (Update) dữ liệu nhật ký    |
|              | cho bất kỳ ai, kể cả tài khoản Root Admin để đảm bảo  |
|              | tính minh bạch kiểm toán cao nhất.                    |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR26-1:** Thao tác \"Ghi log\" vào cơ sở dữ     |
| n-Functional | liệu ở các Use Case khác phải được thiết kế chạy ngầm |
| R            | (Asynchronous) để không làm tăng thời gian phản hồi   |
| equirement** | (Latency) của tiến trình chính.                       |
+--------------+-------------------------------------------------------+

### Use case 27

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC27                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Đăng nhập hệ thống                                    |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người dùng, tôi muốn đăng nhập vào hệ thống    |
| escription** | bằng mật khẩu, mã OTP hoặc qua mạng xã hội (Google,   |
|              | Facebook) để truy cập vào phân quyền làm việc của     |
|              | mình.                                                 |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Mọi tác nhân (Khách hàng, Người bán, Quản trị viên),  |
|              | Hệ thống bên thứ 3 (Google, Facebook).                |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng truy cập trang Đăng nhập và lựa chọn       |
|              | phương thức đăng nhập mong muốn.                      |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người dùng đã có tài khoản trên hệ thống hoặc có tài  |
| ndition(s)** | khoản mạng xã hội (Google/Facebook) đang hoạt động    |
|              | hợp lệ.                                               |
+--------------+-------------------------------------------------------+
| **Post-Co    | Đăng nhập thành công, hệ thống thiết lập phiên làm    |
| ndition(s)** | việc và điều hướng người dùng tới giao diện tương     |
|              | ứng.                                                  |
+--------------+-------------------------------------------------------+
| **Basic      | (Luồng chính: Đăng nhập bằng Email và Mật khẩu truyền |
| Flow**       | thống)                                                |
|              |                                                       |
|              | 1\. Người dùng chọn phương thức \"Đăng nhập bằng Mật  |
|              | khẩu\".                                               |
|              |                                                       |
|              | 2\. Người dùng nhập Email và Mật khẩu đã đăng ký, sau |
|              | đó nhấn \"Đăng nhập\".                                |
|              |                                                       |
|              | 3\. Hệ thống đối chiếu thông tin định danh trong cơ   |
|              | sở dữ liệu.                                           |
|              |                                                       |
|              | 4\. Nếu hợp lệ, hệ thống thiết lập phiên làm việc cho |
|              | người dùng.                                           |
|              |                                                       |
|              | 5\. Hệ thống tự động nhận diện Phân quyền (Role) và   |
|              | điều hướng: Khách hàng về Trang chủ, Người bán/Admin  |
|              | về Bảng điều khiển.                                   |
+--------------+-------------------------------------------------------+
| *            | 1a. Đăng nhập bằng mã OTP (Passwordless):             |
| *Alternative |                                                       |
| Flow**       | 1a1. Người dùng nhập Email và chọn \"Đăng nhập bằng   |
|              | OTP\".                                                |
|              |                                                       |
|              | 1a2. Hệ thống gửi mã xác thực 6 số qua Email người    |
|              | dùng.                                                 |
|              |                                                       |
|              | 1a3. Người dùng nhập mã OTP và xác nhận.              |
|              |                                                       |
|              | 1a4. Hệ thống kiểm tra OTP hợp lệ. *Use Case tiếp tục |
|              | ở Bước 4 của luồng chính.*                            |
|              |                                                       |
|              | 1b. Đăng nhập qua mạng xã hội (Google / Facebook):    |
|              |                                                       |
|              | 1b1. Người dùng chọn nút \"Đăng nhập với Google\"     |
|              | (hoặc Facebook).                                      |
|              |                                                       |
|              | 1b2. Hệ thống chuyển hướng sang màn hình xác thực của |
|              | Google/Facebook.                                      |
|              |                                                       |
|              | 1b3. Người dùng đồng ý cấp quyền truy cập thông tin   |
|              | cơ bản (Email, Tên).                                  |
|              |                                                       |
|              | 1b4. Google/Facebook trả về thông báo xác thực thành  |
|              | công. (Nếu Email này chưa từng tồn tại, hệ thống tự   |
|              | động tạo mới tài khoản Khách hàng).                   |
|              |                                                       |
|              | *Use Case tiếp tục ở Bước 4 của luồng chính.*         |
|              |                                                       |
|              | 1c. Đăng nhập bảo mật 2 lớp (Dành riêng cho Admin):   |
|              |                                                       |
|              | 1c1. Quản trị viên truy cập đường dẫn riêng, nhập     |
|              | Email và Mật khẩu đúng.                               |
|              |                                                       |
|              | 1c2. Hệ thống yêu cầu xác thực bước 2 bằng cách gửi   |
|              | OTP về Email.                                         |
|              |                                                       |
|              | 1c3. Admin nhập đúng OTP, hệ thống cấp phiên làm việc |
|              | vào Admin Dashboard.                                  |
+--------------+-------------------------------------------------------+
| **Exception  | 3a. Sai tài khoản hoặc mật khẩu:                      |
| Flow**       |                                                       |
|              | 3a1. Hệ thống báo lỗi \"Email hoặc mật khẩu không     |
|              | chính xác\" và chặn truy cập.                         |
|              |                                                       |
|              | *Use Case quay lại bước 2.*                           |
|              |                                                       |
|              | 1a3.1 (Ngoại lệ của 1a) Sai OTP:                      |
|              |                                                       |
|              | 1a3.2. Hệ thống báo lỗi \"Mã OTP không hợp lệ hoặc đã |
|              | hết hạn\".                                            |
|              |                                                       |
|              | 1b3.1 (Ngoại lệ của 1b) Từ chối cấp quyền:            |
|              |                                                       |
|              | 1b3.2. Người dùng chọn lệnh \"Hủy\" trên màn hình của |
|              | Google/Facebook.                                      |
|              |                                                       |
|              | 1b3.3. Quá trình đăng nhập thất bại, hệ thống đưa     |
|              | người dùng về lại trang Đăng nhập mặc định.           |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR27-1:** Mã OTP chỉ có hiệu lực sử dụng 1 lần   |
| Rules**      | và tồn tại trong khoảng thời gian giới hạn (Ví dụ: 5  |
|              | phút).                                                |
|              |                                                       |
|              | \- **BR27-2:** Tính năng Đăng nhập qua mạng xã hội    |
|              | (Google/Facebook) mặc định chỉ cấp quyền Khách hàng   |
|              | (Customer).                                           |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR27-1:** Về mặt kỹ thuật, \"Phiên làm việc\"   |
| n-Functional | phải được quản lý bằng chuẩn mã hóa JSON Web Token    |
| R            | (JWT) thông qua bộ lọc Spring Security.               |
| equirement** |                                                       |
|              | \- **NFR27-2:** Mật khẩu truyền thống của người dùng  |
|              | bắt buộc phải được mã hóa một chiều bằng BCrypt khi   |
|              | đối chiếu.                                            |
|              |                                                       |
|              | \- **NFR27-3:** Tính năng đăng nhập Google/Facebook   |
|              | phải được giao tiếp thông qua giao thức chuẩn OAuth2. |
+--------------+-------------------------------------------------------+

### Use case 28

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC28                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Đăng xuất                                             |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người dùng, tôi muốn đăng xuất khỏi hệ thống   |
| escription** | để kết thúc phiên làm việc an toàn, tránh bị người    |
|              | khác truy cập trái phép vào tài khoản của mình.       |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Mọi tác nhân                                          |
+--------------+-------------------------------------------------------+
| **Priority** | Must Have                                             |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng nhấn vào nút \"Đăng xuất\" (Logout) trên   |
|              | màn hình làm việc.                                    |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người dùng đang ở trạng thái đăng nhập hợp lệ.        |
| ndition(s)** |                                                       |
+--------------+-------------------------------------------------------+
| **Post-Co    | Phiên làm việc kết thúc, hệ thống thu hồi quyền truy  |
| ndition(s)** | cập hiện tại của người dùng.                          |
+--------------+-------------------------------------------------------+
| **Basic      | 1\. Người dùng nhấp chọn lệnh \"Đăng xuất\".          |
| Flow**       |                                                       |
|              | 2\. Hệ thống tiếp nhận yêu cầu và tiến hành hủy bỏ    |
|              | phiên làm việc hiện tại trên thiết bị của người dùng. |
|              |                                                       |
|              | 3\. Hệ thống xóa các dữ liệu cá nhân tạm thời đang    |
|              | hiển thị (như Giỏ hàng cá nhân, thông tin Hồ sơ).     |
|              |                                                       |
|              | 4\. Hệ thống tự động điều hướng người dùng trở về     |
|              | Trang chủ mặc định ở trạng thái chưa đăng nhập.       |
+--------------+-------------------------------------------------------+
| *            | *Không có. Luồng xử lý diễn ra trực tiếp một chiều.*  |
| *Alternative |                                                       |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Exception  | *Không có.*                                           |
| Flow**       |                                                       |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR28-1:** Ngay sau khi đăng xuất, mọi liên kết   |
| Rules**      | (URL) riêng tư mà người dùng cố tình truy cập lại (Ví |
|              | dụ: Trang quản lý đơn, trang Checkout) đều phải bị hệ |
|              | thống chặn lại và yêu cầu đăng nhập.                  |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR28-1:** Thao tác đăng xuất phải được xử lý    |
| n-Functional | ngay lập tức tại phía giao diện (Client-side) bằng    |
| R            | cách xóa JWT Token và làm sạch (Clear) Redux Store.   |
| equirement** |                                                       |
+--------------+-------------------------------------------------------+

### Use case 29

+--------------+-------------------------------------------------------+
| **Trường**   | **Nội dung**                                          |
+==============+=======================================================+
| **Use Case   | UC29                                                  |
| ID**         |                                                       |
+--------------+-------------------------------------------------------+
| **Use Case   | Đổi / Quên mật khẩu                                   |
| Name**       |                                                       |
+--------------+-------------------------------------------------------+
| **D          | Là một Người dùng, tôi muốn thiết lập lại mật khẩu    |
| escription** | khi bị quên hoặc chủ động đổi mật khẩu để bảo vệ an   |
|              | toàn cho tài khoản cá nhân.                           |
+--------------+-------------------------------------------------------+
| **Actor(s)** | Mọi tác nhân                                          |
+--------------+-------------------------------------------------------+
| **Priority** | Should Have                                           |
+--------------+-------------------------------------------------------+
| **Trigger**  | Người dùng nhấn vào liên kết \"Quên mật khẩu\" ở màn  |
|              | hình Đăng nhập, hoặc chọn \"Đổi mật khẩu\" trong Quản |
|              | lý tài khoản.                                         |
+--------------+-------------------------------------------------------+
| **Pre-Co     | Người dùng phải sở hữu (truy cập được) vào hòm thư    |
| ndition(s)** | Email đã đăng ký.                                     |
+--------------+-------------------------------------------------------+
| **Post-Co    | Mật khẩu mới được cập nhật thành công vào cơ sở dữ    |
| ndition(s)** | liệu.                                                 |
+--------------+-------------------------------------------------------+
| **Basic      | (Luồng Quên mật khẩu)                                 |
| Flow**       |                                                       |
|              | 1\. Người dùng chọn lệnh \"Quên mật khẩu\" tại màn    |
|              | hình Đăng nhập.                                       |
|              |                                                       |
|              | 2\. Hệ thống yêu cầu cung cấp Email định danh.        |
|              |                                                       |
|              | 3\. Người dùng nhập Email và chọn lệnh \"Gửi mã xác   |
|              | thực\".                                               |
|              |                                                       |
|              | 4\. Hệ thống tra cứu thông tin và gửi mã OTP xác nhận |
|              | về hòm thư Email.                                     |
|              |                                                       |
|              | 5\. Người dùng nhập mã OTP và nhập Mật khẩu mới mong  |
|              | muốn.                                                 |
|              |                                                       |
|              | 6\. Hệ thống xác thực OTP. Nếu hợp lệ, hệ thống tiến  |
|              | hành mã hóa bảo mật mật khẩu mới và ghi đè lên dữ     |
|              | liệu cũ.                                              |
|              |                                                       |
|              | 7\. Hệ thống thông báo cập nhật thành công và đưa     |
|              | người dùng về lại trang Đăng nhập.                    |
+--------------+-------------------------------------------------------+
| *            | 1a. Luồng chủ động Đổi mật khẩu:                      |
| *Alternative |                                                       |
| Flow**       | 1a1. Người dùng đã đăng nhập, truy cập vào tab \"Đổi  |
|              | mật khẩu\" ở trang Hồ sơ.                             |
|              |                                                       |
|              | 1a2. Hệ thống yêu cầu nhập Mật khẩu hiện tại và Mật   |
|              | khẩu mới.                                             |
|              |                                                       |
|              | 1a3. Hệ thống đối chiếu mật khẩu hiện tại. Nếu trùng  |
|              | khớp, hệ thống tiến hành cập nhật mật khẩu mới thành  |
|              | công.                                                 |
|              |                                                       |
|              | *Use Case kết thúc.*                                  |
+--------------+-------------------------------------------------------+
| **Exception  | 4a. Email không tồn tại:                              |
| Flow**       |                                                       |
|              | 4a1. Hệ thống báo lỗi \"Tài khoản Email không tồn     |
|              | tại\" và chặn lệnh gửi mã.                            |
|              |                                                       |
|              | 1a3.1 (Ngoại lệ của 1a): Mật khẩu hiện tại không      |
|              | khớp:                                                 |
|              |                                                       |
|              | 1a3.2. Hệ thống báo lỗi bôi đỏ tại trường nhập liệu   |
|              | và từ chối cập nhật.                                  |
+--------------+-------------------------------------------------------+
| **Business   | \- **BR29-1:** Mật khẩu bắt buộc phải có độ dài tối   |
| Rules**      | thiểu 8 ký tự để đảm bảo tiêu chuẩn an toàn.          |
+--------------+-------------------------------------------------------+
| **No         | \- **NFR29-1:** Mật khẩu mới tuyệt đối không được lưu |
| n-Functional | dưới dạng văn bản thô (Plain-text). Backend (Spring   |
| R            | Boot) bắt buộc phải băm (hash) mật khẩu bằng thuật    |
| equirement** | toán BCrypt trước khi lưu vào Database.               |
+--------------+-------------------------------------------------------+

# THIẾT KẾ DỮ LIỆU

## Sơ đồ logic

### Lược đồ logic

### Chi tiết các bảng dữ liệu

### Ràng buộc toàn vẹn

#### Ràng buộc khóa chính

#### Ràng buộc khóa ngoại

#### Ràng buộc miền giá trị

#### Ràng buộc liên thuộc tính

#### Ràng buộc liên bộ

#### Ràng buộc liên quan hệ

## Sơ đồ cơ sở dữ liệu mức vật lý:

# THIẾT KẾ GIAO DIỆN

## Danh sách các màn hình và sơ đồ chuyển đổi

## Mô tả chi tiết các màn hình

# THIẾT KẾ XỬ LÝ

# CÀI ĐẶT VÀ THỬ NGHIỆM

## Cài đặt chương trình

### Các công cụ hỗ trợ

### Giới thiệu tổng quát về các công nghệ được sử dụng

### Cấu trúc chương trình và quy trình thực hiện

## Kết quả thử nghiệm

### Kết quả tổng quát

### Một số tính năng chạy thực tế

# TỔNG KẾT

## Kết quả đạt được

## Ưu điểm

## Hướng phát triển

**TÀI LIỆU THAM KHẢO**

**BẢNG LIỆT KÊ KHỐI LƯỢNG CÔNG VIỆC**
