import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

const get = (url) => api.get(url).then((r) => r.data);

export const useDatasetStatus = () =>
  useQuery({
    queryKey: ["status"],
    queryFn: () => get("/dataset/status"),
    refetchInterval: (q) => (q.state.data?.ready ? false : 2000),
    staleTime: 0,
  });

export const useReadyQuery = (key, url, enabled = true) => {
  const { data: st } = useDatasetStatus();
  return useQuery({ queryKey: key, queryFn: () => get(url), enabled: !!st?.ready && enabled, staleTime: Infinity });
};

export const useSummary = () => useReadyQuery(["summary"], "/dataset/summary");
export const useFeatureInfo = () => useReadyQuery(["features"], "/features/info");

export const useLatestRun = () =>
  useQuery({
    queryKey: ["run"],
    queryFn: () => get("/experiments/latest"),
    refetchInterval: (q) => {
      const s = q.state.data?.run?.status;
      return s === "running" || s === "queued" ? 3000 : false;
    },
    staleTime: 0,
  });

export const useLoadDataset = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/dataset/load").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
};

export const useRunExperiments = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/experiments/run").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["run"] }),
  });
};
