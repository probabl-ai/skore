<script setup lang="ts">
import { computed, ref } from "vue";

const props = defineProps<{ src: string }>();

const iframe = ref<HTMLIFrameElement>();
const iframeHeight = ref(0);

const document = computed(() => {
  return `
    <!DOCTYPE html>
    <html lang="en">

    <head>
      <meta charset="UTF-8">
      <style>
        * {
          margin: 0;
        }
        body {
          -webkit-font-smoothing: antialiased;
        }
      </style>
    </head>

    <body>${props.src}</body>
  `;
});

function onIframeLoad() {
  if (iframe.value && iframe.value.contentWindow) {
    iframeHeight.value = iframe.value.contentWindow.document.documentElement.scrollHeight;
  }
}
</script>

<template>
  <iframe
    ref="iframe"
    :srcdoc="document"
    frameborder="0"
    sandbox="allow-scripts allow-same-origin"
    scrolling="no"
    @load="onIframeLoad"
    :style="{ height: `${iframeHeight}px` }"
  ></iframe>
</template>

<style scoped>
iframe {
  width: 100%;
}
</style>
