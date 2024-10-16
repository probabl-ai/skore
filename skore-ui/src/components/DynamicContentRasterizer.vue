<script setup lang="ts">
import { debounce } from "@/services/utils";
import { toPng } from "html-to-image";
import { computed, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

const props = defineProps<{
  isRasterized: boolean;
}>();

const rasterizedSrc = ref("");
const realContent = useTemplateRef("realContent");
const contentHeight = ref(0);

const styles = computed(() => {
  const h = contentHeight.value !== 0 ? `${contentHeight.value}px` : "auto";
  return {
    height: h,
  };
});

const updateCache = debounce(
  async () => {
    if (realContent.value) {
      rasterizedSrc.value = await toPng(realContent.value as HTMLElement);
    }
  },
  100,
  false
);

const resizeObserver = new ResizeObserver(async (entries) => {
  if (entries.length == 1 && !props.isRasterized) {
    const content = entries[0];
    contentHeight.value = content.contentRect.height;
    updateCache();
  }
});

watch(
  () => realContent.value,
  () => {
    if (realContent.value) {
      resizeObserver.observe(realContent.value as HTMLElement);
    }
  }
);

onBeforeUnmount(() => {
  resizeObserver.disconnect();
});
</script>

<template>
  <div class="dynamic-content-rasterizer" :style="styles">
    <div class="real-content" v-if="!isRasterized" ref="realContent">
      <slot />
    </div>
    <div class="cached-content" v-else>
      <img :src="rasterizedSrc" />
    </div>
  </div>
</template>

<style scoped>
.dynamic-content-rasterizer {
  position: relative;
  padding: 0;
  margin: 0;

  & .rasterized-content {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
  }
}
</style>
