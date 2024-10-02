<script setup lang="ts">
import Simplebar from "simplebar-vue";

import ReportCanvas from "@/components/ProjectViewCanvas.vue";
</script>

<template>
  <div class="share">
    <div class="share-header">
      <h1>Report</h1>
    </div>
    <Simplebar class="canvas-wrapper">
      <ReportCanvas :showCardActions="false" />
    </Simplebar>
  </div>
</template>

<style scoped>
@media (prefers-color-scheme: dark) {
  main {
    --editor-placeholder-image: url("./assets/images/editor-placeholder-dark.svg");
  }
}

@media (prefers-color-scheme: light) {
  main {
    --editor-placeholder-image: url("./assets/images/editor-placeholder-light.svg");
  }
}

.share {
  display: flex;
  height: 100dvh;
  max-height: 100vh;
  flex-direction: column;
  flex-grow: 1;

  & .share-header {
    display: flex;
    height: 44px;
    align-items: center;
    padding: var(--spacing-padding-large);
    border-right: solid var(--border-width-normal) var(--border-color-normal);
    border-bottom: solid var(--border-width-normal) var(--border-color-normal);
    background-color: var(--background-color-elevated);

    & h1 {
      flex-grow: 1;
      color: var(--text-color-normal);
      font-size: var(--text-size-title);
      font-weight: var(--text-weight-title);
      text-align: center;
    }
  }

  & .drop-indicator {
    height: 3px;
    border-radius: 8px;
    margin: 0 10%;
    background-color: var(--text-color-title);
    opacity: 0;
    transition: opacity var(--transition-duration) var(--transition-easing);

    &.visible {
      opacity: 1;
    }
  }

  & .placeholder {
    height: 100%;
    padding-top: calc((100vh - var(--header-height)) * 476 / 730);
    background-image: var(--editor-placeholder-image);
    background-position: 50%;
    background-repeat: no-repeat;
    background-size: contain;
    color: var(--text-color-normal);
    font-size: var(--text-size-normal);
    text-align: center;
  }

  & .canvas-wrapper {
    height: 0;
    flex-grow: 1;
    padding: var(--spacing-padding-large);
  }
}
</style>
