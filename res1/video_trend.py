import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 시각화 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rc('font', family='Malgun Gothic')

class GameSalesAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df_cleaned = None
    
    def preprocess_data(self):
        """데이터 전처리"""
        # 원본 데이터 복사
        df_a = self.df.copy()
        
        # 불필요한 열 제거
        df_a = df_a.drop(self.df.iloc[:,0].name, axis=1)
        
        # 판매량 데이터 변환
        sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        for col in sales_columns:
            df_a[col] = df_a[col].apply(self._convert_to_number)
        
        # 총 판매량 계산
        df_a['Total_Sales'] = df_a[sales_columns].sum(axis=1)
        
        # 연도 데이터 정제
        df_a = self._clean_years(df_a)
        
        # 결측치 처리
        df_a = self._handle_missing_values(df_a)
        
        # 연도 그룹 생성
        df_a['Year_Group'] = df_a['Year'] // 10 * 10
        
        self.df_cleaned = df_a
        return df_a

    @staticmethod
    def _convert_to_number(value):
        """판매량 문자열을 숫자로 변환"""
        if 'K' in str(value):
            return float(str(value).replace('K', '')) / 1000
        elif 'M' in str(value):
            return float(str(value).replace('M', ''))
        return float(value)

    @staticmethod
    def _clean_years(df):
        """연도 데이터 정제"""
        df.loc[df['Year'] <= 20, 'Year'] += 2000
        df.loc[(df['Year'] < 100) & (df['Year'] > 60), 'Year'] += 1900
        return df

    def _handle_missing_values(self, df):
        """결측치 처리"""
        # Year 결측치 처리
        df = df.dropna(subset=['Year'])
        
        # Genre 결측치 처리
        df.loc[df['Name'] == 'Pokemon X/Pokemon Y', 'Genre'] = 'Role-Playing'
        df.loc[df['Name'] == 'Wii Party', 'Genre'] = 'Misc'
        df.loc[df['Name'] == 'Final Fantasy XII', 'Genre'] = 'Role-Playing'
        df = df.dropna(subset=['Genre'])
        
        # Publisher 결측치 처리
        df.loc[df['Name'] == 'wwe Smackdown vs. Raw 2006', 'Publisher'] = 'THQ'
        df = df.dropna(subset=['Publisher'])
        
        return df

    def analyze_genre_sales(self):
        """장르별 판매량 분석"""
        # 장르별 총 판매량
        genre_sales = self.df_cleaned.groupby('Genre')['Total_Sales'].sum().reset_index()
        genre_sales = genre_sales.sort_values(by='Total_Sales', ascending=False)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Total_Sales', y='Genre', data=genre_sales, palette='viridis')
        ax.set(xlabel='Total Sales (millions)', ylabel='Genre', title='Total Sales by Genre')
        plt.tight_layout()
        plt.show()

    def analyze_regional_sales(self):
        """지역별 판매량 분석"""
        sales_by_region = self.df_cleaned.groupby('Genre')[
            ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        ].sum().reset_index()

        plt.figure(figsize=(14, 8))
        sales_melted = sales_by_region.melt(
            id_vars='Genre', 
            var_name='Region', 
            value_name='Sales'
        )
        sns.lineplot(data=sales_melted, x='Genre', y='Sales', hue='Region')
        plt.title('Regional Sales by Genre')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_platform_lifecycle(self):
        """플랫폼 수명주기 분석"""
        platform_years = self.df_cleaned.groupby(['Platform', 'Year'])['Total_Sales'].sum().reset_index()
        
        plt.figure(figsize=(15, 8))
        for platform in self.df_cleaned['Platform'].unique():
            data = platform_years[platform_years['Platform'] == platform]
            plt.plot(data['Year'], data['Total_Sales'], label=platform)
        
        plt.title('Platform Sales Over Time')
        plt.xlabel('Year')
        plt.ylabel('Total Sales')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def create_ml_features(self):
        """머신러닝을 위한 특성 생성"""
        df_features = self.df_cleaned.copy()
        
        # 연도별 평균 판매량
        year_avg_sales = df_features.groupby('Year')['Total_Sales'].transform('mean')
        df_features['Year_Avg_Sales'] = year_avg_sales
        
        # 장르별 평균 판매량
        genre_avg_sales = df_features.groupby('Genre')['Total_Sales'].transform('mean')
        df_features['Genre_Avg_Sales'] = genre_avg_sales
        
        return df_features

    def train_ml_model(self):
        """머신러닝 모델 학습 및 평가"""
        features = ['Platform', 'Genre', 'Publisher', 'Year']
        X = pd.get_dummies(self.df_cleaned[features])
        y = self.df_cleaned['Total_Sales']

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 모델 파이프라인 생성
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])

        # 모델 학습 및 평가
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # 교차 검증
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 메인 실행 코드
if __name__ == "__main__":
    # 분석기 초기화
    analyzer = GameSalesAnalyzer('C:\\Users\\Jo\\python\\BOJ\\res1\\vgames2.csv')
    
    # 데이터 전처리
    analyzer.preprocess_data()
    
    # 분석 실행
    analyzer.analyze_genre_sales()
    analyzer.analyze_regional_sales()
    analyzer.analyze_platform_lifecycle()
    
    # 머신러닝 모델 학습
    analyzer.train_ml_model()
