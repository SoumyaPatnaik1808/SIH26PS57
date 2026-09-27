# ==============================================================================
# Multi-stage production build for Next.js frontend
# ==============================================================================
FROM node:20-alpine AS base

# Install libc compatibility layer if needed for Alpine
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Step 1: Install dependencies based on package-lock.json
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm ci

# Step 2: Build the production application
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Accept client-side backend API URL at build time (defaults to localhost:8000 for browser client access)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# Step 3: Minimal production runner
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy static assets and standalone server output
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
