# Chrome이 최신 버전임
# 버전 131.0.6778.86(공식 빌드) (64비트)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.service import Service

import pandas as pd
import re
import time

# ChromeDriver 경로 설정
driver_path = "C:\\Users\\Jo\\Downloads\\chromedriver-win64\\chromedriver.exe"

# Service 객체 생성
service = Service(driver_path)

# Chrome WebDriver 설정
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(service=service, options=options)
driver.get("https://map.naver.com/v5/search")

print(driver.title)  # 현재 페이지 제목 출력

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.input_search")))

# 검색어 입력 및 검색
search_box = driver.find_element(By.CSS_SELECTOR, "input.input_search")
search_box.send_keys("강남역 맛집")
search_box.send_keys(Keys.ENTER)

# iframe 전환
WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "searchIframe")))

# 스크롤 부분

scroll_div = driver.find_element(By.CSS_SELECTOR,"#_pcmap_list_scroll_container")
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#_pcmap_list_scroll_container")))

for _ in range(6):
    try:
        scroll_script = "arguments[0].scrollBy(0,3000);"
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#_pcmap_list_scroll_container")))
        driver.execute_script(scroll_script, scroll_div)
    except Exception as e:
        print("스크롤 요소를 찾을 수 없습니다.", e)
        
button_number = 1
data = []


while True:
    
    result_items = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".CHC5F"))
        )
    for result_item in result_items:
        name = result_item.find_element(By.CSS_SELECTOR, ".place_bluelink.N_KDL .TYaxT").text
        try:
            rating_text = result_item.find_element(By.CSS_SELECTOR, ".h69bs.orXYY").text
            rating = float(re.search(r'\d+(\.\d+)?', rating_text).group())
        except NoSuchElementException:
            rating = 0.0
        reviews_text = result_item.find_element(By.CSS_SELECTOR, ".MVx6e").text
        reviews = int(re.search(r'리뷰 (\d+)', reviews_text).group(1)) if re.search(r'리뷰 (\d+)', reviews_text) else 0
        
        if reviews >= 999:
            result_item.find_element(By.CSS_SELECTOR, ".tzwk0").send_keys(Keys.ENTER)
            driver.switch_to.default_content()
            WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "entryIframe")))

            # 상세 리뷰 데이터 수집
            reviews_999 = driver.find_elements(By.CSS_SELECTOR, ".PXMot")
            reviews_text = [x.text for x in reviews_999]
            
            # reviews_text는 한 장소의 리뷰 데이터를 담고 있는 문자열
            reviews = sum(
                int(line.split()[2].replace(',', ''))
                for line in reviews_text 
                if "리뷰" in line and "별점" not in line  # "리뷰" 포함, "별점" 제외
            )




            driver.switch_to.default_content()
            try:
                iframe = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "searchIframe"))
                )
                driver.switch_to.frame(iframe)
            except TimeoutException:
                print("searchIframe 로드 시간 초과")
            except Exception as e:
                print("searchIframe 처리 중 오류:", e)

        data.append({"장소 이름": name, "별점": rating, "리뷰수": reviews})
    
    # XPath로 버튼 요소 동적 생성
    button_number += 1
    button_xpath = f"//*[@id='app-root']/div/div[2]/div[2]/a[{button_number+1}]"

    try:
        # "다음" 버튼 찾기
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, button_xpath))
        )

        # "다음페이지"인 경우 종료
        if button.text.strip() == "다음페이지":
            print("마지막 페이지입니다. 루프를 종료합니다.")
            break

        # 버튼 클릭
        button.click()
        
        time.sleep(1)

       

    except NoSuchElementException:
        print("다음 버튼을 찾을 수 없습니다. 루프를 종료합니다.")
        break




print(data)

# WebDriver 종료
driver.quit()
