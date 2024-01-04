# Description

보건복지부가 발간한 [나에게 힘이 되는 복지서비스](https://m.bokjiro.go.kr/ssis-tem/twatxe/wlfarePr/selectWlfareBrochure.do) 책자를 Data로 이용하였습니다.\
전체 책자는 주제 - 상세 제도로 나누어져 있으며 구성은 다음과 같습니다.


1. 생계 지원
2. 취업 지원
3. 임신 보육 지원
4. 청소년 청년 지원
5. 보건의료 지원
6. 노령층 지원
7. 장애인 지원
8. 보훈대상자 지원
9. 법률 금융 복지 지원
10. 기타 위기별 상황별 지원

해당 주제들의 세부 제도를 Embedding model의 max_sequence에 따라 Token 개수로 Split하였고,\
이를 Chroma DB에 저장하여 RAG 시스템에서 Retrieve 할 수 있도록 하였습니다.