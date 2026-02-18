# Hyundai Senior Silson Mock Documents

이 폴더는 `hyundai_senior_silson_denial_sample.pdf` 케이스를 기준으로 만든
`DOCUMENT_CATALOG` 대응 mock 문서 세트입니다.

## 구성
- `manifest.json`: catalog id -> mock 파일 경로 매핑
- 각 `*.md`: 해당 catalog 문서의 샘플 본문

## 사용 방법
1. `manifest.json`에서 필요한 `catalog id`의 파일 경로를 찾습니다.
2. `request_additional_documents` 인터럽트에서 요구된 `required_document_ids` 순서대로
   해당 파일 경로 목록을 만들어 `resume`에 넣습니다.

예시:
- required_document_ids: `["denial_notice", "medical_certificate", "billing_statement"]`
- resume paths:
  1. `data/mock_documents/hyundai_senior_silson_case/denial_notice.md`
  2. `data/mock_documents/hyundai_senior_silson_case/medical_certificate.md`
  3. `data/mock_documents/hyundai_senior_silson_case/billing_statement.md`
