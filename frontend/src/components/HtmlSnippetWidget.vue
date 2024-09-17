<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";

interface Props {
  src?: string;
  base64Src?: string;
}
const props = withDefaults(defineProps<Props>(), {
  src: "",
  base64Src: "",
});

const iframe = ref<HTMLIFrameElement>();
const iframeHeight = ref(0);

const src = computed(() => {
  return props.src || btoa(props.base64Src);
});
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

    <body>${src.value}</body>
  `;
});

const resizeObserver = new ResizeObserver((entries) => {
  if (entries.length == 1) {
    const observedIframe = entries[0];
    iframeHeight.value = observedIframe.contentRect.height;
  }
});

function onIframeLoad() {
  if (iframe.value && iframe.value.contentDocument) {
    resizeObserver.observe(iframe.value.contentDocument.body);
  }
}

onBeforeUnmount(() => {
  if (iframe.value && iframe.value.contentDocument) {
    resizeObserver.unobserve(iframe.value.contentDocument.body);
  }
});
</script>

<template>
  <iframe
    ref="iframe"
    :srcdoc="document"
    frameborder="0"
    sandbox="allow-scripts allow-same-origin"
    scrolling="auto"
    @load="onIframeLoad"
    :style="{ height: `${iframeHeight}px` }"
  ></iframe>
</template>

<style scoped>
iframe {
  width: 100%;
}
</style>
