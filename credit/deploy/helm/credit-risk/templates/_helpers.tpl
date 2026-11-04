{{- define "credit-risk.name" -}}
{{- .Release.Name | trunc 50 | trimSuffix "-" -}}
{{- end -}}

{{- define "credit-risk.labels" -}}
app.kubernetes.io/name: credit-risk
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/* Khung Pod dùng chung cho bản ổn định (stable) và bản canary */}}
{{- define "credit-risk.podSpec" -}}
containers:
  - name: api
    image: "{{ .image }}"
    imagePullPolicy: {{ .root.Values.image.pullPolicy }}
    ports:
      - name: http
        containerPort: {{ .root.Values.service.targetPort }}
    env:
      - name: MODEL_DIR
        value: /app/models
    readinessProbe:
      httpGet: { path: {{ .root.Values.probes.readiness.path }}, port: http }
      initialDelaySeconds: {{ .root.Values.probes.readiness.initialDelaySeconds }}
      periodSeconds: 5
    livenessProbe:
      httpGet: { path: {{ .root.Values.probes.liveness.path }}, port: http }
      initialDelaySeconds: {{ .root.Values.probes.liveness.initialDelaySeconds }}
      periodSeconds: 10
    resources:
      {{- toYaml .root.Values.resources | nindent 6 }}
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
      allowPrivilegeEscalation: false
{{- end -}}
