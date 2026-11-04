// Kiểm thử tải bằng k6 (miễn phí): k6 run deploy/k6/load.js
// Đổi địa chỉ: k6 run -e BASE_URL=http://localhost:8080 deploy/k6/load.js
import http from "k6/http";
import { check, sleep } from "k6";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";

export const options = {
  stages: [
    { duration: "30s", target: 10 },  // tăng dần lên 10 người dùng ảo
    { duration: "1m", target: 50 },   // tải cao: quan sát HPA tăng số Pod (kubectl get hpa -w)
    { duration: "30s", target: 0 },   // giảm tải
  ],
  thresholds: {
    http_req_failed: ["rate<0.01"],     // dưới 1% lỗi
    http_req_duration: ["p(95)<300"],   // 95% yêu cầu dưới 300 ms (mục tiêu, chỉnh theo máy bạn)
  },
};

const application = {
  CNT_CHILDREN: 1, AMT_INCOME_TOTAL: 180000, AMT_CREDIT: 600000, AMT_ANNUITY: 30000,
  AMT_GOODS_PRICE: 540000, DAYS_BIRTH: -12000, DAYS_EMPLOYED: -2000, DAYS_ID_PUBLISH: -3000,
  EXT_SOURCE_1: 0.5, EXT_SOURCE_2: 0.6, EXT_SOURCE_3: 0.55, REGION_RATING_CLIENT: 2,
  NAME_CONTRACT_TYPE: "Cash loans", FLAG_OWN_CAR: "N", FLAG_OWN_REALTY: "Y",
  NAME_INCOME_TYPE: "Working", NAME_EDUCATION_TYPE: "Secondary / secondary special",
  NAME_FAMILY_STATUS: "Married", NAME_HOUSING_TYPE: "House / apartment", OCCUPATION_TYPE: "Laborers",
};

export default function () {
  // Thay đổi thu nhập một chút để mỗi yêu cầu khác nhau
  const body = { ...application, AMT_INCOME_TOTAL: 100000 + Math.floor(Math.random() * 200000) };
  const res = http.post(`${BASE_URL}/predict`, JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
  check(res, {
    "status 200": (r) => r.status === 200,
    "có quyết định": (r) => ["approve", "reject"].includes(r.json("decision")),
  });
  sleep(0.2);
}
