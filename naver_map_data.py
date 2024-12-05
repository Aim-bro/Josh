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
# 고정된 ChromeDriver 경로
default_driver_path = "C:\\Users\\Jo\\Downloads\\chromedriver-win64\\chromedriver.exe"

def initialize_driver(driver_path = default_driver_path):
    """
    WebDriver를 초기화하고 설정을 반환합니다.
    """
    service = Service(driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    # options.add_argument('--headless') # 사용시 오류 발생


    driver = webdriver.Chrome(service=service, options=options)
    return driver

def navigate_to_page(driver, url):
    """
    특정 URL로 이동합니다.
    """
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.input_search")))

def search_place(driver, keyword):
    """
    검색창에 키워드를 입력하고 검색합니다.
    """
    search_box = driver.find_element(By.CSS_SELECTOR, "input.input_search")
    search_box.send_keys(keyword)
    search_box.send_keys(Keys.ENTER)
    WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "searchIframe")))

def scroll_results(driver, scroll_count=6):
    """
    검색 결과를 스크롤합니다.
    """
    scroll_div = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#_pcmap_list_scroll_container")))
    for _ in range(scroll_count):
        try:
            driver.execute_script("arguments[0].scrollBy(0, 3000);", scroll_div)
        except Exception as e:
            print("스크롤 실패:", e)
            break

def extract_place_data(driver):
    """
    현재 페이지의 장소 데이터를 추출합니다.
    """
    data = []
    result_items = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".CHC5F")))

    for result_item in result_items:
        name = result_item.find_element(By.CSS_SELECTOR, ".place_bluelink.N_KDL .TYaxT").text
        try:
            rating_text = result_item.find_element(By.CSS_SELECTOR, ".h69bs.orXYY").text
            rating = float(re.search(r'\d+(\.\d+)?', rating_text).group())
        except NoSuchElementException:
            rating = 0.0
        reviews_text = result_item.find_element(By.CSS_SELECTOR, ".MVx6e").text
        reviews = int(re.search(r'리뷰 (\d+)', reviews_text).group(1)) if re.search(r'리뷰 (\d+)', reviews_text) else 0
        data.append({"장소 이름": name, "별점": rating, "리뷰수": reviews})
    return data

def click_next_button(driver, button_number):
    """
    다음 버튼을 클릭하고 페이지를 넘어갑니다.
    """
    button_xpath = f"//*[@id='app-root']/div/div[2]/div[2]/a[{button_number+1}]"
    try:
        button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, button_xpath)))
        if button.text.strip() == "다음페이지":
            return False
        button.click()
        time.sleep(1)
        return True
    except NoSuchElementException:
        return False

def scrape_places(keyword, scroll_count=6):
    """
    네이버 지도에서 장소 데이터를 스크래핑합니다.
    """
    driver = initialize_driver()
    try:
        navigate_to_page(driver, "https://map.naver.com/v5/search")
        search_place(driver, keyword)
        scroll_results(driver, scroll_count)
        
        data = []
        button_number = 1

        while True:
            data.extend(extract_place_data(driver))
            if not click_next_button(driver, button_number):
                break
            button_number += 1
        return data
    finally:
        driver.quit()



# keyword = "을지로 맛집"
# data = scrape_places(keyword)

# # 데이터 출력
# print(data)
