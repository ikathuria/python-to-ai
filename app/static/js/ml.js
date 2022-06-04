$(".model-choice").change(function () {
  $(".show-reg-div").hide();
  $("show-reg").removeAttr("required");
  $(".show-reg").removeAttr("data-error");

  log.console($(this).val());
  if ($(this).val() == "Regression") {
    $(".show-reg-div").show();
    $(".show-reg").attr("required", "");
    $(".show-reg").attr("data-error", "This field is required.");
  } else {
    $(".show-reg-div").hide();
    $("show-reg").removeAttr("required");
    $(".show-reg").removeAttr("data-error");
  }
});

$(".model-choice").trigger("change");
