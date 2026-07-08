{{/*
Expand the name of the chart.
*/}}
{{- define "hermes-agentic-rl.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "hermes-agentic-rl.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "hermes-agentic-rl.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "hermes-agentic-rl.labels" -}}
helm.sh/chart: {{ include "hermes-agentic-rl.chart" . }}
{{ include "hermes-agentic-rl.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "hermes-agentic-rl.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hermes-agentic-rl.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "hermes-agentic-rl.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hermes-agentic-rl.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
ConfigMap name for training config
*/}}
{{- define "hermes-agentic-rl.configMapName" -}}
{{- printf "%s-config" (include "hermes-agentic-rl.fullname" .) }}
{{- end }}

{{/*
PVC name for outputs / checkpoints
*/}}
{{- define "hermes-agentic-rl.pvcName" -}}
{{- printf "%s-data" (include "hermes-agentic-rl.fullname" .) }}
{{- end }}

{{/*
Generate the container command based on workload type and command settings
*/}}
{{- define "hermes-agentic-rl.containerCommand" -}}
{{- $cmd := .Values.command }}
{{- $configPath := "/etc/hermes/config.yaml" }}
{{- $outputPath := "/data/outputs" }}
- python
- -m
- hermes_agentic_rl.cli.main
- {{ $cmd }}
- --config
- {{ $configPath }}
{{- if or (eq $cmd "train-rl") (eq $cmd "eval-rl") (eq $cmd "eval-gate") (eq $cmd "benchmark-suite") }}
- --output
- {{ $outputPath }}
{{- end }}
{{- if .Values.extraArgs }}
{{- range .Values.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}
