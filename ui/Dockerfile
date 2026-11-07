# Production-ready, security-hardened Dockerfile for RAHBAR UI
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Copy custom unprivileged nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy single-page application static HTML
COPY index.html /usr/share/nginx/html/index.html

# Explicit non-root user (UID 101 nginx)
USER 101

EXPOSE 8080

# Exec form ensures signals reach nginx PID 1 for graceful shutdown
CMD ["nginx", "-g", "daemon off;"]

