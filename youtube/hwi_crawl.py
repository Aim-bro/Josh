from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
import pandas as pd
import time

# .env 불러오기
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")

# 유튜브 API 클라이언트 생성
yt = build("youtube", "v3", developerKey=API_KEY)

# 최근 1년간의 유튜브 목록 가져오기

def get_youtube_videos(channel_id : str) -> str:
    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError("No channel found")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_videos(upload_playlist_id : str, n : int = 50) -> pd.DataFrame:
    r = yt.playlistItems().list(
        part="contentDetails,snippet",
        playlistId=upload_playlist_id,
        maxResults=min(n,50)
    ).execute()
    
    out = []
    for it in r.get("items", []):
        cd, sn = it["contentDetails"], it["snippet"]
        out.append({
            "video_id": cd["videoId"],
            "published_at": pd.to_datetime(cd.get("videoPublishedAt"), utc=True),
            "title": sn.get("title"),
            "description": sn.get("description"),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df

# 댓글 크롤링
from googleapiclient.errors import HttpError
from datetime import datetime, timezone

def comments_of_video(video_id: str) -> pd.DataFrame:
    rows, token = [], None
    while True:
        try:
            resp = yt.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order='time',
                pageToken=token,
                textFormat="plainText"
            ).execute()
        except HttpError as e:
            print(f"[WARN] skip video {video_id}: {e}")
            break
        
        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            top_id = it["snippet"]["topLevelComment"]["id"]
            rows.append({
                "comment_id": top_id,
                "video_id": video_id,
                "author": top.get("authorDisplayName"),
                "text": top.get("textDisplay") or top.get("textOriginal"),
                "like_count": top.get("likeCount", 0),
                "published_at": top.get("publishedAt"),
                "reply_count": it["snippet"].get("totalReplyCount", 0),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.05)
    
    if not rows:
        return pd.DataFrame(columns=[
            "comment_id", "video_id", "author", "text", "like_count", "published_at", "reply_count"])
    
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    for col in ("like_count", "reply_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df

def save_csv(df: pd.DataFrame, video_id: str) -> str:
    os.makedirs("data/comments", exist_ok=True)   # data/comments 폴더 생성
    path = f"data/comments/{video_id}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def main ():
    uploads_id = get_youtube_videos(CHANNEL_ID)
    df_videos = list_videos(uploads_id, n=50)
    os.makedirs("data/meta", exist_ok=True)
    df_videos.to_csv("data/meta/hwi_videos_list.csv", index=False, encoding="utf-8-sig")
    for i, row in df_videos.iterrows():
        vid = row["video_id"]
        print(f"[FETCH] {i+1}/{len(df_videos)} {vid} - {row['title']}")
        df_comments = comments_of_video(vid)
        p = save_csv(df_comments, vid)
        print(f"  -> saved {len(df_comments)} comments to {p}")


if __name__ == "__main__":
    main()