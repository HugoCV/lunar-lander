import axios from "axios";
const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1";
export const api = axios.create({ baseURL: API_BASE, timeout: 15000 });

export type VideoItem = {
  name: string;
  url: string; // server-relative, e.g. /videos/file.mp4
  sizeBytes: number;
  modified: number; // epoch seconds
};

export async function listVideos(): Promise<VideoItem[]> {
  const { data } = await api.get("/train/videos");
  return data.videos as VideoItem[];
}
