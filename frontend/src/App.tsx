import { useEffect, useMemo, useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  Stack,
  Chip,
  Snackbar,
  Alert,
  Link,
  Paper,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Skeleton,
  Divider,
  Tooltip as MuiTooltip,
  LinearProgress,
  alpha,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import VideocamIcon from "@mui/icons-material/Videocam";
import AssessmentIcon from "@mui/icons-material/Assessment";
import HealthAndSafetyIcon from "@mui/icons-material/HealthAndSafety";
import MemoryIcon from "@mui/icons-material/Memory";
import SportsScoreIcon from "@mui/icons-material/SportsScore";
import TimelineIcon from "@mui/icons-material/Timeline";
import WhatshotIcon from "@mui/icons-material/Whatshot";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import VideoGallery from "./components/VideoGallery";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1";
const api = axios.create({ baseURL: API_BASE, timeout: 15000 });

// ——— Types ———
export type TrainStatus = {
  running: boolean;
  hasAgent: boolean;
  episodes: number;
  lastReward: number | null;
  epsilon: number | null;
  memorySize: number;
  historyTail: [number, number][];
};

// ——— API ———
async function getHealth() {
  const { data } = await api.get("/health");
  return data;
}
async function getStatus(): Promise<TrainStatus> {
  const { data } = await api.get("/train/status");
  return data;
}
async function startTraining() {
  const { data } = await api.post("/train/start");
  return data;
}
async function stopTraining() {
  const { data } = await api.post("/train/stop");
  return data;
}
async function evaluateAgent() {
  const { data } = await api.post("/train/evaluate");
  return data as { ok: boolean; mean: number; scores: number[] };
}
async function recordVideo() {
  const { data } = await api.post("/train/video");
  return data as { ok: boolean; path: string | null };
}

// ——— Utils ———
const fmt = (n: number | null | undefined, digits = 1, fallback = "—") =>
  n != null && Number.isFinite(n) ? n.toFixed(digits) : fallback;

function StatCard({
  label,
  value,
  icon,
  help,
}: {
  label: string;
  value: string | number | JSX.Element;
  icon?: JSX.Element;
  help?: string;
}) {
  const theme = useTheme();
  return (
    <Card
      variant="outlined"
      sx={{
        height: "100%",
        borderRadius: 3,
        borderColor: alpha(theme.palette.primary.main, 0.25),
        bgcolor: alpha(theme.palette.primary.main, 0.04),
        transition: "transform .15s ease, box-shadow .15s ease",
        "&:hover": { transform: "translateY(-2px)", boxShadow: 6 },
      }}
    >
      <CardContent sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
        {icon && (
          <Box
            aria-hidden
            sx={{
              p: 1,
              borderRadius: 2,
              bgcolor: alpha(theme.palette.primary.main, 0.1),
            }}
          >
            {icon}
          </Box>
        )}
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="caption" color="text.secondary" noWrap>
            {label}
          </Typography>
          <MuiTooltip title={help ?? ""} disableHoverListener={!help} arrow>
            <Typography variant="h5" fontWeight={700} mt={0.5} noWrap>
              {value}
            </Typography>
          </MuiTooltip>
        </Box>
      </CardContent>
    </Card>
  );
}

function RewardChart({
  episodes,
  rewards,
  loading,
}: {
  episodes: number[];
  rewards: number[];
  loading: boolean;
}) {
  const data = useMemo(
    () => episodes.map((ep, i) => ({ ep, reward: rewards[i] })),
    [episodes, rewards]
  );
  const hasData = data.length > 0;

  return (
    <Paper
      variant="outlined"
      sx={{ p: 2, borderRadius: 3, position: "relative", overflow: "hidden" }}
    >
      {loading && (
        <LinearProgress sx={{ position: "absolute", inset: 0, height: 2 }} />
      )}

      <Stack direction="row" alignItems="center" spacing={1} mb={1.5}>
        <AssessmentIcon fontSize="small" />
        <Typography variant="subtitle1" fontWeight={700}>
          Rewards (Last episodes)
        </Typography>
      </Stack>
      <Divider sx={{ mb: 2 }} />
      {hasData ? (
        <Box sx={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 24, left: 4, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ep" tickMargin={6} />
              <YAxis />
              <Tooltip
                formatter={(v: number) => v.toFixed(2)}
                labelFormatter={(l: number) => `Episodio ${l}`}
              />
              <Line type="monotone" dataKey="reward" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      ) : (
        <Box textAlign="center" py={6}>
          <Typography variant="body2" color="text.secondary">
            Sin datos aún
          </Typography>
        </Box>
      )}
    </Paper>
  );
}

export default function App() {
  const theme = useTheme();
  const isMdUp = useMediaQuery(theme.breakpoints.up("md"));

  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [toast, setToast] = useState<{
    open: boolean;
    msg: string;
    severity: "success" | "info" | "warning" | "error";
  }>({ open: false, msg: "", severity: "info" });

  // — Poll status every 2s —
  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const s = await getStatus();
        if (mounted) setStatus(s);
      } catch {
        /* ignore */
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const apiOrigin = useMemo(() => {
    try {
      return new URL(API_BASE).origin;
    } catch {
      return window.location.origin;
    }
  }, []);

  const videoUrl = useMemo(() => {
    if (!videoPath) return null;
    return videoPath.startsWith("http") ? videoPath : `${apiOrigin}${videoPath}`;
  }, [videoPath, apiOrigin]);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
  }, []);

  // — Actions —
  const onStart = async () => {
    setLoading(true);
    try {
      await startTraining();
      setToast({ open: true, msg: "Entrenamiento iniciado", severity: "success" });
    } catch (e: any) {
      setToast({ open: true, msg: e?.message || "Error al iniciar", severity: "error" });
    } finally {
      setLoading(false);
    }
  };

  const onStop = async () => {
    setLoading(true);
    try {
      await stopTraining();
      setToast({ open: true, msg: "Señal de stop enviada", severity: "info" });
    } catch (e: any) {
      setToast({ open: true, msg: e?.message || "Error al detener", severity: "error" });
    } finally {
      setLoading(false);
    }
  };

  const onEvaluate = async () => {
    setLoading(true);
    try {
      const res = await evaluateAgent();
      if (res.ok) {
        setToast({
          open: true,
          msg: `Evaluación OK. Mean=${res.mean.toFixed(1)} (${res.scores
            .map((s) => s.toFixed(1))
            .join(", ")})`,
          severity: "success",
        });
      } else {
        setToast({ open: true, msg: "No hay agente aún para evaluar", severity: "warning" });
      }
    } catch (e: any) {
      setToast({ open: true, msg: e?.message || "Error en evaluación", severity: "error" });
    } finally {
      setLoading(false);
    }
  };

  const onVideo = async () => {
    setLoading(true);
    try {
      const res = await recordVideo();
      if (res.ok && res.path) {
        const backendOrigin = new URL(
          import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1"
        ).origin;
        const url = backendOrigin + res.path; // e.g. http://localhost:8000/videos/file.mp4
        setVideoPath(url);
        setToast({ open: true, msg: `Video recorded: ${url}`, severity: "success" });
      } else {
        setToast({ open: true, msg: "Could not record video (agent not ready?)", severity: "warning" });
      }
    } catch (e: any) {
      setToast({ open: true, msg: e?.message || "Error recording video", severity: "error" });
    } finally {
      setLoading(false);
    }
  };

  const episodes = status?.historyTail?.map(([ep]) => ep) ?? [];
  const rewards = status?.historyTail?.map(([_, r]) => r) ?? [];
  const loadingStatus = status == null;

  const lastReward = status?.lastReward ?? null;
  const rewardTrend = useMemo(() => {
    const recent = rewards.slice(-10);
    if (recent.length < 2) return 0;
    return recent[recent.length - 1] - recent[0];
  }, [rewards]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh", bgcolor: "background.default" }}>
      {/* Top App Bar with subtle gradient */}
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          bgcolor: "transparent",
          borderBottom: 1,
          borderColor: "divider",
          backgroundImage: (theme) =>
            `linear-gradient(180deg, ${alpha(theme.palette.background.paper, 0.9)} 0%, ${theme.palette.background.paper} 50%, ${theme.palette.background.default} 100%)`,
          backdropFilter: "saturate(1.2) blur(6px)",
        }}
      >
        <Toolbar>
          <HealthAndSafetyIcon sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1 }} fontWeight={800}>
            Lunar Lander RL Dashboard
          </Typography>
          <Chip
            size="small"
            icon={<WhatshotIcon fontSize="small" />}
            label={health ? "Backend OK" : "Sin conexión"}
            color={health ? "success" : "warning"}
            variant={health ? "filled" : "outlined"}
          />
        </Toolbar>
      </AppBar>

      <Container component="main" maxWidth="lg" sx={{ py: 3, flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ width: '100%' }}>
          {/* Action Panel */}
          <Paper
          variant="outlined"
          sx={{
            p: 2,
            mb: 2,
            borderRadius: 3,
            background: (t) => alpha(t.palette.primary.main, 0.035),
          }}
        >
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "stretch", md: "center" }}
          >
            <Stack direction="row" spacing={1} flexWrap="wrap">
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={onStart}
                disabled={loading || !!status?.running}
              >
                Start Training
              </Button>
              <Button
                variant="outlined"
                color="warning"
                startIcon={<StopIcon />}
                onClick={onStop}
                disabled={loading || !status?.running}
              >
                Stop
              </Button>
              <Button
                variant="outlined"
                startIcon={<AssessmentIcon />}
                onClick={onEvaluate}
                disabled={loading}
              >
                Evaluate
              </Button>
              <Button
                variant="outlined"
                startIcon={<VideocamIcon />}
                onClick={onVideo}
                disabled={loading}
              >
                Record Video
              </Button>
              {loading && <CircularProgress size={22} sx={{ ml: 1 }} />}
            </Stack>

            {videoUrl && (
              <Paper sx={{ p: 1, ml: { md: 2 }, borderRadius: 2 }}>
                <Typography variant="body2">
                  Último video:&nbsp;
                  <Link href={videoUrl} underline="hover" target="_blank" rel="noreferrer">
                    {videoUrl}
                  </Link>
                </Typography>
                <Box mt={1} sx={{ width: isMdUp ? 480 : "100%" }}>
                  <video src={videoUrl} controls width="100%" style={{ borderRadius: 8 }} />
                </Box>
              </Paper>
            )}
          </Stack>
        </Paper>

        {/* Stats */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Card variant="outlined" sx={{ height: "100%", borderRadius: 3 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Status
                </Typography>
                {loadingStatus ? (
                  <Stack direction="row" spacing={1} mt={1}>
                    <Skeleton variant="rounded" width={80} height={28} />
                    <Skeleton variant="rounded" width={110} height={28} />
                  </Stack>
                ) : (
                  <Stack direction="row" spacing={1} mt={1}>
                    <Chip
                      label={status?.running ? "Training" : "Idle"}
                      color={status?.running ? "warning" : "default"}
                    />
                    <Chip
                      label={status?.hasAgent ? "Agent Ready" : "No Agent"}
                      color={status?.hasAgent ? "success" : "default"}
                    />
                  </Stack>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={6} md={2}>
            <StatCard
              label="Episodes"
              value={
                loadingStatus ? (
                  <Skeleton width={60} />
                ) : (
                  status?.episodes ?? 0
                )
              }
              icon={<TimelineIcon />}
            />
          </Grid>
          <Grid item xs={6} md={2}>
            <StatCard
              label="Last Reward"
              value={loadingStatus ? <Skeleton width={60} /> : fmt(lastReward, 1)}
              help={typeof lastReward === "number" ? `${lastReward.toFixed(3)}` : undefined}
              icon={<SportsScoreIcon />}
            />
          </Grid>
          <Grid item xs={6} md={2}>
            <StatCard
              label="Epsilon"
              value={loadingStatus ? <Skeleton width={60} /> : fmt(status?.epsilon, 3)}
              icon={<WhatshotIcon />}
            />
          </Grid>
          <Grid item xs={6} md={2}>
            <StatCard
              label="Memory"
              value={loadingStatus ? <Skeleton width={60} /> : status?.memorySize ?? 0}
              icon={<MemoryIcon />}
            />
          </Grid>
        </Grid>

        {/* Trend chip */}
        <Box mt={1} mb={1}>
          <Chip
            size="small"
            variant="outlined"
            color={rewardTrend >= 0 ? "success" : "error"}
            label={`Tendencia (10 eps): ${rewardTrend >= 0 ? "+" : ""}${fmt(rewardTrend, 1)}`}
          />
        </Box>

        {/* Video (full-width) */}
        {videoUrl && (
          <Paper sx={{ p: 2, borderRadius: 3, mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Último video
            </Typography>
            <Box sx={{ aspectRatio: "16 / 9", width: "100%" }}>
              <video
                src={videoUrl}
                controls
                preload="metadata"
                style={{ width: "100%", height: "100%", borderRadius: 8, display: "block" }}
              />
            </Box>
          </Paper>
        )}

        {/* Chart */}
        <Box mt={2}>
          <RewardChart episodes={episodes} rewards={rewards} loading={loadingStatus} />
        </Box>

        {/* Gallery */}
        <Box mt={3}>
          <VideoGallery />
        </Box>
        </Box>
      </Container>

      {/* Toast */}
      <Snackbar
        open={toast.open}
        autoHideDuration={4000}
        onClose={() => setToast((t) => ({ ...t, open: false }))}
      >
        <Alert
          onClose={() => setToast((t) => ({ ...t, open: false }))}
          severity={toast.severity}
          sx={{ width: "100%" }}
        >
          {toast.msg}
        </Alert>
      </Snackbar>
    </Box>
  );
}
