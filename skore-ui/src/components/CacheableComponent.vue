<script setup lang="ts">
import { debounce } from "@/services/utils";
import { toPng } from "html-to-image";
import { computed, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

const props = defineProps<{
  isCached: boolean;
}>();

const cachedSrc = ref("");
const realContent = useTemplateRef("realContent");
const cacheHeight = ref(0);

const styles = computed(() => {
  const h = cacheHeight.value !== 0 ? `${cacheHeight.value}px` : "auto";
  return {
    height: h,
  };
});

const updateCache = debounce(
  async () => {
    if (realContent.value) {
      cachedSrc.value = await toPng(realContent.value as HTMLElement);
    }
  },
  100,
  false
);

const resizeObserver = new ResizeObserver(async (entries) => {
  if (entries.length == 1 && !props.isCached) {
    const content = entries[0];
    cacheHeight.value = content.contentRect.height;
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
  <div class="cacheable" :style="styles">
    <div class="real-content" v-if="!isCached" ref="realContent">
      <slot />
    </div>
    <div class="cached-content" v-else>
      <img :src="cachedSrc" />
    </div>
  </div>
</template>

<style scoped>
.cacheable {
  position: relative;
  padding: 0;
  margin: 0;

  & .cached-content {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
  }
}
</style>
