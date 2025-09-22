import { useEffect, useMemo, useState } from "react";
import { Box, Paper, Typography, List, ListItemButton, ListItemText, Divider, Link } from "@mui/material";
import { listVideos, type VideoItem } from "../api";

function bytesToHuman(b: number) {
  const u = ["B","KB","MB","GB","TB"];
  let i = 0; let x = b;
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
  return `${x.toFixed(1)} ${u[i]}`;
}

export default function VideoGallery() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [selected, setSelected] = useState<VideoItem | null>(null);

  const backendOrigin = useMemo(
    () => new URL(import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1").origin,
    []
  );

  useEffect(() => {
    (async () => {
      const v = await listVideos();
      setVideos(v);
      if (v.length > 0) setSelected(v[0]);
    })();
  }, []);

  const videoUrl = selected ? backendOrigin + selected.url : "";

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "320px 1fr" }, gap: 2, mt: 3 }}>
      <Paper variant="outlined" sx={{ maxHeight: 420, overflow: "auto" }}>
        <List dense>
          {videos.map((v) => (
            <div key={v.name}>
              <ListItemButton selected={selected?.name === v.name} onClick={() => setSelected(v)}>
                <ListItemText
                  primary={v.name}
                  secondary={`${bytesToHuman(v.sizeBytes)} • ${new Date(v.modified * 1000).toLocaleString()}`}
                />
              </ListItemButton>
              <Divider component="li" />
            </div>
          ))}
        </List>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        {selected ? (
          <>
            <Typography variant="subtitle1" gutterBottom>
              {selected.name} —{" "}
              <Link href={videoUrl} target="_blank" rel="noreferrer">open</Link>
            </Typography>
            <video src={videoUrl} controls width="100%" />
          </>
        ) : (
          <Typography>No videos available.</Typography>
        )}
      </Paper>
    </Box>
  );
}
