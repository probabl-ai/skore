(function () {
  const clientID = "460c80a5e0b7c04";

  const script = document.createElement("script");
  script.src = "https://static.reo.dev/" + clientID + "/reo.js";
  script.defer = true;
  script.onload = function () {
      Reo.init({ clientID });
  };

  document.head.appendChild(script);
})();
