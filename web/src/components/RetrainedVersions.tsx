"use client";

import { useCallback, useEffect, useState } from "react";
import {
  type ModelVersionsResponse,
  type RetrainedVersion,
  api,
  relativeTime,
} from "@/lib/api";
import { Card, Skeleton } from "./ui";

export function RetrainedVersions({
  modelId,
  onVersionPromoted,
}: {
  modelId: string;
  onVersionPromoted?: (version: string) => void;
}) {
  const [data, setData] = useState<ModelVersionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const load = useCallback(async () => {
    const res = await api.modelVersions(modelId);
    if (res.data) {
      setData(res.data);
    } else if (res.error) {
      setError(res.error);
    }
    setLoading(false);
  }, [modelId]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    load();
  }, [load]);

  const handleTriggerRetrain = async () => {
    setRetraining(true);
    setError(null);
    const res = await api.triggerRetrain(modelId);
    if (res.data) {
      await load();
      setExpandedId(res.data.id);
    } else {
      setError(res.error || "Failed to trigger retraining.");
    }
    setRetraining(false);
  };

  const [rollingBack, setRollingBack] = useState<string | null>(null);

  const handlePromote = async (version: string) => {
    setPromoting(version);
    setError(null);
    const res = await api.promoteVersion(modelId, version);
    if (res.data) {
      await load();
      onVersionPromoted?.(version);
    } else {
      setError(res.error || "Failed to promote version.");
    }
    setPromoting(null);
  };

  const handleRollback = async (version?: string) => {
    const key = version || "latest";
    setRollingBack(key);
    setError(null);
    const res = await api.rollbackVersion(modelId, version);
    if (res.data) {
      await load();
      onVersionPromoted?.(res.data.current_version);
    } else {
      setError(res.error || "Failed to execute rollback.");
    }
    setRollingBack(null);
  };

  if (loading && !data) {
    return (
      <Card className="p-5">
        <Skeleton className="h-6 w-48 mb-3" />
        <Skeleton className="h-28 w-full" />
      </Card>
    );
  }

  const versions = data?.versions || [];
  const currentVersion = data?.current_version || "1.0.0";
  const isBlocked = data?.retrain_blocked;
  const isRecommended = data?.retrain_recommended;

  return (
    <Card className="overflow-hidden">
      {/* ---- header ---- */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-5 py-3.5">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-fg">
              Retrained Versions & Challengers
            </span>
            <span className="rounded border border-border bg-bg px-2 py-0.5 font-mono text-2xs text-muted">
              Active: {currentVersion}
            </span>
            <span className="rounded border border-border bg-bg px-2 py-0.5 font-mono text-2xs text-subtle">
              {versions.length} versions
            </span>
            {versions.length > 1 && (
              <button
                onClick={() => handleRollback()}
                disabled={rollingBack !== null}
                className="inline-flex items-center gap-1 rounded border border-amber-600/30 bg-amber-500/10 px-2 py-0.5 text-2xs font-medium text-amber-700 transition hover:bg-amber-500/20 dark:text-amber-400 disabled:opacity-50"
                title="Roll back active production state to previous stable version"
              >
                <svg
                  className="h-3 w-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="2"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3"
                  />
                </svg>
                {rollingBack === "latest" ? "Rolling back…" : "Roll Back"}
              </button>
            )}
          </div>
          <p className="mt-0.5 text-2xs text-subtle">
            Champion vs. Challenger evaluations judged by Gemma on the drifted slice
          </p>
        </div>

        <div className="flex items-center gap-2">
          {isRecommended && (
            <button
              onClick={handleTriggerRetrain}
              disabled={retraining}
              className="inline-flex items-center gap-1.5 rounded-md bg-fg px-3 py-1.5 text-xs font-medium text-bg shadow-sm transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {retraining ? (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-bg opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-bg" />
                  </span>
                  Evaluating with Gemma…
                </>
              ) : (
                <>
                  <svg
                    className="h-3.5 w-3.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth="2"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
                    />
                  </svg>
                  Trigger Retraining
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* ---- error message ---- */}
      {error && (
        <div className="border-b border-border bg-[#dc26260d] px-5 py-2.5 text-xs text-retrain">
          {error}
        </div>
      )}

      {/* ---- policy block banner for pipeline faults ---- */}
      {isBlocked && (
        <div className="border-b border-border bg-[#ca8a040d] px-5 py-4">
          <div className="flex items-start gap-3">
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#ca8a0422] text-xs font-bold text-investigate">
              !
            </span>
            <div className="flex-1">
              <h4 className="text-xs font-semibold text-fg">
                Retraining Blocked by Arbiter Policy
              </h4>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                {data?.block_reason}
              </p>
              <div className="mt-2.5 flex items-center gap-2">
                <span className="rounded border border-[#ca8a0433] bg-[#ca8a0415] px-2 py-0.5 text-2xs font-medium text-investigate">
                  Action: INVESTIGATE upstream
                </span>
                <span className="text-2xs text-subtle">
                  Training on corrupt data would bake the bug into the model
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---- retrained versions list ---- */}
      <div className="divide-y divide-border">
        {versions.length === 0 && !isBlocked ? (
          <div className="px-5 py-8 text-center text-xs text-muted">
            <p>No retrained versions recorded for this model yet.</p>
            <p className="mt-1 text-2xs text-subtle">
              When drift is detected and Arbiter diagnoses covariate shift, retrained challenger
              candidates are trained and evaluated here.
            </p>
          </div>
        ) : (
          versions.map((v) => {
            const isPromoted = v.status === "PROMOTED";
            const isChallenger = v.status === "CHALLENGER";
            const isExpanded = expandedId === v.id;

            return (
              <div
                key={v.id}
                className={`transition-colors ${
                  isPromoted ? "bg-surface/50" : "bg-bg hover:bg-surface/20"
                } p-5`}
              >
                {/* top metadata row */}
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-sm font-semibold tracking-tight text-fg">
                      {v.version}
                    </span>

                    {/* status pill */}
                    <span
                      className={`inline-flex items-center rounded-full px-2 py-0.5 font-mono text-2xs font-semibold uppercase tracking-wider ${
                        isPromoted
                          ? "border border-green-600/30 bg-green-500/10 text-green-600 dark:text-green-400"
                          : isChallenger
                            ? "border border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400"
                            : "border border-border bg-surface text-muted"
                      }`}
                    >
                      {v.status}
                    </span>

                    {/* recommendation badge */}
                    <span
                      className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-2xs font-medium ${
                        v.recommendation === "PROMOTE"
                          ? "border-green-600/20 bg-green-500/10 text-green-700 dark:text-green-300"
                          : v.recommendation === "REJECT"
                            ? "border-red-600/20 bg-red-500/10 text-retrain"
                            : "border-yellow-600/20 bg-yellow-500/10 text-investigate"
                      }`}
                    >
                      <span>Gemma:</span>
                      <strong className="font-semibold">{v.recommendation}</strong>
                      <span className="text-2xs opacity-70">
                        ({Math.round(v.confidence * 100)}%)
                      </span>
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="text-2xs text-subtle">
                      {relativeTime(v.created_at)}
                    </span>

                    {isChallenger && v.recommendation === "PROMOTE" && (
                      <button
                        onClick={() => handlePromote(v.version)}
                        disabled={promoting === v.version}
                        className="rounded border border-green-600/40 bg-green-600/10 px-2.5 py-1 text-2xs font-medium text-green-700 transition hover:bg-green-600/20 dark:text-green-300 disabled:opacity-50"
                      >
                        {promoting === v.version ? "Promoting…" : "Promote to Champion"}
                      </button>
                    )}

                    {!isPromoted && (
                      <button
                        onClick={() => handleRollback(v.version)}
                        disabled={rollingBack === v.version}
                        className="inline-flex items-center gap-1 rounded border border-amber-600/40 bg-amber-500/10 px-2.5 py-1 text-2xs font-medium text-amber-700 transition hover:bg-amber-500/20 dark:text-amber-300 disabled:opacity-50"
                        title={`Roll back active production model to ${v.version}`}
                      >
                        <svg
                          className="h-3 w-3"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth="2"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3"
                          />
                        </svg>
                        {rollingBack === v.version ? "Rolling back…" : "Roll Back"}
                      </button>
                    )}

                    {isPromoted && (
                      <span className="inline-flex items-center gap-1 font-mono text-2xs text-green-600 dark:text-green-400">
                        <svg
                          className="h-3 w-3"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 0 1 0 1.414l-8 8a1 1 0 0 1-1.414 0l-4-4a1 1 0 0 1 1.414-1.414L8 12.586l7.293-7.293a1 1 0 0 1 1.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Champion
                      </span>
                    )}
                  </div>
                </div>

                {/* trigger cause */}
                {v.trigger_reason && (
                  <div className="mt-2 text-xs text-muted">
                    <span className="text-subtle font-medium">Trigger: </span>
                    <span>{v.trigger_reason}</span>
                    {v.triggered_by_event && (
                      <span className="ml-1.5 font-mono text-2xs text-subtle">
                        [{v.triggered_by_event}]
                      </span>
                    )}
                  </div>
                )}

                {/* metrics comparison box */}
                <div className="mt-3.5 grid grid-cols-2 gap-3 sm:grid-cols-4 rounded-lg border border-border bg-surface p-3">
                  <div>
                    <span className="text-2xs uppercase tracking-wider text-subtle">
                      Baseline Accuracy
                    </span>
                    <div className="mt-0.5 font-mono text-xs font-semibold text-fg">
                      {(v.champion_metric * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <span className="text-2xs uppercase tracking-wider text-subtle">
                      Retrained Accuracy
                    </span>
                    <div className="mt-0.5 flex items-baseline gap-1.5">
                      <span className="font-mono text-xs font-semibold text-fg">
                        {(v.challenger_metric * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`font-mono text-2xs ${
                          v.overall_delta >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-retrain"
                        }`}
                      >
                        {v.overall_delta >= 0 ? "+" : ""}
                        {(v.overall_delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <span className="text-2xs uppercase tracking-wider text-subtle truncate block" title={v.segment_name}>
                      Drifted Slice Delta
                    </span>
                    <div className="mt-0.5 flex items-baseline gap-1.5">
                      <span className="font-mono text-xs font-semibold text-fg">
                        {v.challenger_segment_metric !== undefined && v.challenger_segment_metric !== null
                          ? `${(v.challenger_segment_metric * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                      {v.segment_delta !== undefined && v.segment_delta !== null && (
                        <span
                          className={`font-mono text-2xs font-bold ${
                            v.segment_delta >= 0
                              ? "text-green-600 dark:text-green-400"
                              : "text-retrain"
                          }`}
                        >
                          {v.segment_delta >= 0 ? "+" : ""}
                          {(v.segment_delta * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>

                  <div>
                    <span className="text-2xs uppercase tracking-wider text-subtle">
                      Training Window
                    </span>
                    <div className="mt-0.5 font-mono text-xs text-muted">
                      {v.training_samples?.toLocaleString() || "1,200"} rows
                    </div>
                  </div>
                </div>

                {/* Gemma's evaluation reason */}
                {v.reasoning && (
                  <div className="mt-3 rounded-md border border-border/80 bg-bg px-3.5 py-2.5">
                    <div className="flex items-center gap-1.5 text-2xs font-medium text-subtle">
                      <span>Reasoning from Gemma</span>
                      {v.source && <span className="font-mono">({v.source})</span>}
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      {v.reasoning}
                    </p>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}
